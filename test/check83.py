import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import gc
import threading
import cv2
import time
import argparse
import sys
import os
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class RealSense3DScanner:
    def __init__(self, voxel_size=0.02, downsample_voxel_size=0.005,
                 sdf_trunc=0.04, max_fps=30, cuda_device="CUDA:0",
                 output_dir="output", visualize=False, max_fragments=100):
        """
        Initializes the RealSense3DScanner object with optimized parameters.

        :param voxel_size: The voxel size used for downsampling the point cloud during scanning.
        :param downsample_voxel_size: The voxel size used for final downsampling of the saved point cloud.
        :param sdf_trunc: The truncation distance for TSDF integration.
        :param max_fps: Maximum frames per second for RealSense streams.
        :param cuda_device: The device to be used for operations (e.g., "CUDA:0" or "CPU:0").
        :param output_dir: Directory where output files will be saved.
        :param visualize: Boolean indicating whether to visualize the scanning process in real-time.
        :param max_fragments: Maximum number of fragments to store to limit memory usage.
        """
        # Initialize parameters
        self.voxel_size = voxel_size
        self.downsample_voxel_size = downsample_voxel_size
        self.sdf_trunc = sdf_trunc
        self.max_fps = max_fps
        self.device = o3d.core.Device(cuda_device) if cuda_device.startswith("CUDA") else o3d.core.Device(cuda_device)
        self.fragments = []
        self.rgbd_frames = []
        self.pose_graph = o3d.pipelines.registration.PoseGraph()
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.intrinsic = None
        self.output_dir = output_dir
        self.visualize = visualize
        self.vis = None
        self.vis_thread = None
        self.should_stop_visualizer = threading.Event()
        self.max_fragments = max_fragments  # To limit memory usage

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.output_dir, 'scanner.log'))
            ]
        )

        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Lock for thread-safe operations
        self.lock = threading.Lock()

        # Event to signal stopping the scanning loop
        self.stop_event = threading.Event()

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        """
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            # Optimize RealSense stream profiles for performance
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.max_fps)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.max_fps)

            profile = self.pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                intrinsics.width,
                intrinsics.height,
                intrinsics.fx,
                intrinsics.fy,
                intrinsics.ppx,
                intrinsics.ppy)
            logging.info("Camera intrinsics retrieved successfully.")
        except RuntimeError as e:
            logging.error(f"Failed to start pipeline: {e}")
            if hasattr(self, 'pipeline') and self.pipeline:
                device = self.pipeline.get_active_profile().get_device()
                device.hardware_reset()
                self.pipeline.stop()
            sys.exit(1)

    def capture_rgbd_frame(self):
        """
        Captures and returns an RGBD frame from the RealSense camera.

        :return: A tuple (color_image, depth_image) as NumPy arrays, or (None, None) if frames are unavailable.
        """
        try:
            # Wait for frames with a timeout and check for stop_event
            frames = self.pipeline.poll_for_frames()
            if not frames:
                time.sleep(0.01)
                return None, None

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                logging.warning("Incomplete frames received.")
                return None, None

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image
        except RuntimeError as e:
            logging.error(f"Failed to capture frames: {e}")
            return None, None

    def capture_point_cloud(self):
        """
        Captures and processes a point cloud from the RealSense camera.

        :return: A tuple (processed_point_cloud, RGBDImage) or (None, None) if capture fails.
        """
        color_image, depth_image = self.capture_rgbd_frame()

        if color_image is None or depth_image is None:
            return None, None

        try:
            depth_image_o3d = o3d.geometry.Image(depth_image)
            color_image_o3d = o3d.geometry.Image(color_image)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_o3d,
                depth_image_o3d,
                depth_scale=1.0 / self.depth_scale,
                convert_rgb_to_intensity=False)
            with self.lock:
                self.rgbd_frames.append(rgbd)
        except Exception as e:
            logging.error(f"Failed to create RGBD image: {e}")
            return None, None

        try:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.intrinsic)
            pcd.transform([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])  # Align to Open3D coordinate system

            pcd_down = pcd.voxel_down_sample(self.voxel_size)
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=20))

            return pcd_down, rgbd
        except Exception as e:
            logging.error(f"Failed to generate point cloud: {e}")
            return None, None

    def register_fragments(self):
        """
        Registers all captured fragments using RGB-D odometry.
        Constructs and optimizes a pose graph for the entire sequence.
        """
        n_fragments = len(self.rgbd_frames)
        if n_fragments == 0:
            logging.warning("No RGBD frames to register.")
            return

        odometry = np.identity(4)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry.copy()))
        for i in range(1, n_fragments):
            source_rgbd = self.rgbd_frames[i - 1]
            target_rgbd = self.rgbd_frames[i]

            logging.info(f"Registering frame {i - 1} to frame {i} using RGB-D odometry.")

            try:
                odo_init = np.identity(4)
                option = o3d.pipelines.odometry.OdometryOption()
                [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                    source_rgbd, target_rgbd, self.intrinsic, odo_init,
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                if success:
                    odometry = np.dot(trans, odometry)
                    self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    self.pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(i - 1, i, trans, info, uncertain=False))
                    logging.debug(f"Odometry estimation successful for frame {i}.")
                else:
                    logging.warning(f"Odometry estimation failed between frame {i - 1} and frame {i}.")
                    # Use identity transform
                    self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                    # Add zero information edge
                    info = np.zeros((6, 6))
                    self.pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(i - 1, i, np.identity(4), info, uncertain=True))
            except Exception as e:
                logging.error(f"Odometry estimation failed: {e}")
                continue

        # Pose Graph Optimization
        logging.info("Optimizing pose graph...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 15,
            edge_prune_threshold=0.25,
            preference_loop_closure=0.1,
            reference_node=0)

        try:
            o3d.pipelines.registration.global_optimization(
                self.pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
            logging.info("Pose graph optimization complete.")
        except Exception as e:
            logging.error(f"Pose graph optimization failed: {e}")

    def integrate_fragments_into_tsdf(self):
        """
        Integrates all RGBD frames into the TSDF volume using the optimized poses.
        """
        logging.info("Integrating RGBD frames into TSDF volume...")
        if len(self.rgbd_frames) != len(self.pose_graph.nodes):
            logging.warning("Mismatch between RGBD frames and pose graph nodes. Integration may be inaccurate.")

        for i, (rgbd, node) in enumerate(zip(self.rgbd_frames, self.pose_graph.nodes)):
            logging.debug(f"Integrating frame {i}")
            extrinsic = node.pose

            if not np.all(np.isfinite(extrinsic)):
                logging.warning(f"Frame {i} has invalid extrinsic matrix. Skipping integration.")
                continue

            try:
                self.tsdf_volume.integrate(rgbd, self.intrinsic, np.linalg.inv(extrinsic))
            except Exception as e:
                logging.error(f"Failed to integrate frame {i}: {e}")
                continue
        logging.info("TSDF integration complete.")

    def extract_mesh_from_tsdf(self):
        """
        Extracts a mesh from the integrated TSDF volume.
        """
        logging.info("Extracting mesh from TSDF volume...")
        try:
            mesh = self.tsdf_volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()

            vertices = np.asarray(mesh.vertices)
            if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
                logging.warning("Mesh contains invalid vertices. Removing them...")
                valid_idx = ~np.isnan(vertices).any(axis=1) & ~np.isinf(vertices).any(axis=1)
                mesh = mesh.select_by_index(np.where(valid_idx)[0])
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                logging.info("Invalid vertices removed.")

            if len(mesh.vertices) == 0:
                logging.error("Extracted mesh has no valid vertices.")
                return None

            logging.info("Mesh extraction complete.")
            return mesh
        except Exception as e:
            logging.error(f"Mesh extraction failed: {e}")
            return None

    def start_scanning(self):
        """
        Starts the 3D scanning process in a separate thread.
        """
        scan_thread = threading.Thread(target=self.scanning_loop)
        scan_thread.start()
        logging.info("Scanning started. Press Enter to stop...")
        input()
        self.stop_event.set()
        scan_thread.join()
        self.pipeline.stop()
        logging.info("Scanning stopped.")

    def scanning_loop(self):
        """
        Scanning loop that continuously captures point clouds from the camera.
        """
        fragment_count = 0
        last_time = time.time()
        while not self.stop_event.is_set():
            pcd_frame, rgbd = self.capture_point_cloud()

            if pcd_frame and len(pcd_frame.points) > 0:
                logging.debug(f"Captured fragment {fragment_count} with {len(pcd_frame.points)} points.")
                with self.lock:
                    self.fragments.append(pcd_frame)
                    # No need to append rgbd here, it's done in capture_point_cloud()
                    fragment_count += 1

                    # Limit the number of fragments to manage memory
                    if len(self.fragments) > self.max_fragments:
                        removed_pcd = self.fragments.pop(0)
                        del removed_pcd

                    if len(self.rgbd_frames) > self.max_fragments:
                        with self.lock:
                            removed_rgbd = self.rgbd_frames.pop(0)
                            del removed_rgbd

                if self.visualize:
                    self.update_visualization(pcd_frame)
            else:
                logging.debug("No valid point cloud captured, skipping frame.")

            # Calculate and log frame rate
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed > 1.0:
                fps = fragment_count / elapsed
                logging.info(f"Current FPS: {fps:.2f}")
                last_time = current_time
                fragment_count = 0

            # Sleep briefly to yield control and check for stop_event
            time.sleep(0.001)

            # Invoke garbage collection selectively
            if fragment_count % 10 == 0:
                gc.collect()

    def update_visualization(self, pcd):
        """
        Updates the real-time visualization with the latest point cloud fragment.

        :param pcd: The latest point cloud fragment to visualize.
        """
        if self.vis is None:
            self.initialize_visualization()

        if self.vis is not None:
            with self.lock:
                self.vis.add_geometry(pcd)
            # Limit the number of geometries to prevent slowdowns
            if len(self.vis.get_geometries()) > self.max_fragments:
                self.vis.remove_geometry(self.vis.get_geometries()[0], reset_bounding_box=False)
            self.vis.poll_events()
            self.vis.update_renderer()

    def initialize_visualization(self):
        """
        Initializes the Open3D visualizer in a separate thread.
        """
        def visualize_thread():
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Real-Time 3D Scanning", width=800, height=600)
            self.vis.get_render_option().background_color = np.asarray([0, 0, 0])
            self.vis.get_render_option().point_size = 2
            while not self.should_stop_visualizer.is_set():
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.01)
            self.vis.destroy_window()

        self.vis_thread = threading.Thread(target=visualize_thread)
        self.vis_thread.start()

    def finalize_visualization(self):
        """
        Finalizes and closes the Open3D visualizer.
        """
        if self.visualize and self.vis is not None:
            self.should_stop_visualizer.set()
            self.vis_thread.join()
            self.vis = None

    def save_mesh(self, mesh, filename="output_mesh.ply"):
        """
        Saves the reconstructed mesh to a file.

        :param mesh: The mesh to save.
        :param filename: Filename for the mesh with colors.
        """
        try:
            mesh_filepath = os.path.join(self.output_dir, filename)
            # Ensure the mesh has color information
            if not mesh.has_vertex_colors():
                logging.warning("Mesh does not have vertex colors. Colors will not be saved.")
            o3d.io.write_triangle_mesh(mesh_filepath, mesh, write_vertex_colors=True)
            logging.info(f"Saved mesh with colors to {mesh_filepath}")
        except Exception as e:
            logging.error(f"Failed to save mesh: {e}")

    def run(self):
        """
        Main function that orchestrates the 3D scanning, processing, and saving pipeline.
        """
        start_time = datetime.now()
        logging.info(f"3D Scanning started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            self.start_pipeline()
            if self.visualize:
                self.initialize_visualization()
            self.start_scanning()
            self.register_fragments()
            self.integrate_fragments_into_tsdf()
            mesh = self.extract_mesh_from_tsdf()
            if mesh:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mesh_filename = f"output_mesh_{timestamp}.ply"
                self.save_mesh(mesh, filename=mesh_filename)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Exiting...")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        finally:
            self.finalize_visualization()
            self.executor.shutdown(wait=True)
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"3D Scanning ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Total duration: {duration}")

def parse_arguments():
    """
    Parses command-line arguments for configuration flexibility.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RealSense 3D Scanner with Open3D")
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="Voxel size for downsampling (default: 0.02)")
    parser.add_argument("--downsample_voxel_size", type=float, default=0.005,
                        help="Voxel size for final downsampling (default: 0.005)")
    parser.add_argument("--sdf_trunc", type=float, default=0.04,
                        help="Truncation distance for TSDF (default: 0.04)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for RealSense streams (default: 30)")
    parser.add_argument("--cuda_device", type=str, default="CUDA:0",
                        help='Device for operations, e.g., "CUDA:0" or "CPU:0" (default: "CUDA:0")')
    parser.add_argument("--output_dir", type=str, default="output",
                        help='Directory to save output files (default: "output")')
    parser.add_argument("--visualize", action='store_true',
                        help="Enable real-time visualization (default: False)")
    parser.add_argument("--max_fragments", type=int, default=100,
                        help="Maximum number of fragments to store to limit memory usage (default: 100)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    scanner = RealSense3DScanner(
        voxel_size=args.voxel_size,
        downsample_voxel_size=args.downsample_voxel_size,
        sdf_trunc=args.sdf_trunc,
        max_fps=args.fps,
        cuda_device=args.cuda_device,
        output_dir=args.output_dir,
        visualize=args.visualize,
        max_fragments=args.max_fragments
    )
    scanner.run()

