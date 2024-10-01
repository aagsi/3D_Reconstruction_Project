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
from queue import Queue

class RealSense3DScanner:
    def __init__(self, voxel_size=0.001, downsample_voxel_size=0.001,
                 sdf_trunc=0.005, max_fps=30, cuda_device="CUDA:0",
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
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            device=self.device)
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

        # Initialize thread-safe queue for frame processing
        self.frame_queue = Queue(maxsize=10)

        # Event to signal stopping the scanning loop
        self.stop_event = threading.Event()

        # Initialize frame index for saving files
        self.frame_index = 0
        self.total_frames = 0

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

            # Initialize RealSense depth filters
            self.decimation_filter = rs.decimation_filter()
            self.spatial_filter = rs.spatial_filter()
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling_filter = rs.hole_filling_filter()
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
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if not frames:
                return None, None

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                logging.warning("Incomplete frames received.")
                return None, None

            # Apply filters to depth frame
            depth_frame = self.decimation_filter.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame)

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

            # Save color and depth images to files
            color_filename = os.path.join(self.output_dir, f"color_{self.frame_index:05d}.png")
            depth_filename = os.path.join(self.output_dir, f"depth_{self.frame_index:05d}.png")
            o3d.io.write_image(color_filename, color_image_o3d)
            o3d.io.write_image(depth_filename, depth_image_o3d)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.intrinsic)
            pcd.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])  # Align to Open3D coordinate system

            # Save point cloud to file
            pcd_filename = os.path.join(self.output_dir, f"pcd_{self.frame_index:05d}.ply")
            o3d.io.write_point_cloud(pcd_filename, pcd)

            return pcd, rgbd
        except Exception as e:
            logging.error(f"Failed to generate point cloud: {e}")
            return None, None

    def processing_loop(self):
        """
        Processing loop that handles frame processing and TSDF integration.
        """
        odometry = np.identity(4)
        prev_rgbd = None

        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                pcd_frame, rgbd = self.frame_queue.get(timeout=1)

                if rgbd is not None:
                    if prev_rgbd is not None:
                        # Estimate odometry between frames
                        option = o3d.pipelines.odometry.OdometryOption()
                        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                            prev_rgbd, rgbd, self.intrinsic, np.identity(4),
                            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

                        if success:
                            odometry = np.dot(odometry, np.linalg.inv(trans))
                        else:
                            logging.warning("Odometry estimation failed, using previous pose.")

                    # Integrate into TSDF volume
                    self.tsdf_volume.integrate(rgbd, self.intrinsic, np.linalg.inv(odometry))

                    prev_rgbd = rgbd

                    if self.visualize and pcd_frame is not None:
                        self.update_visualization(pcd_frame)

                # Frame rate calculation
                self.frame_index += 1

            except Exception as e:
                logging.error(f"Error in processing loop: {e}")
                continue

    def start_scanning(self):
        """
        Starts the 3D scanning process in separate threads.
        """
        capture_thread = threading.Thread(target=self.scanning_loop)
        process_thread = threading.Thread(target=self.processing_loop)
        capture_thread.start()
        process_thread.start()
        logging.info("Scanning started. Press Enter to stop...")
        input()
        self.stop_event.set()
        capture_thread.join()
        process_thread.join()
        self.pipeline.stop()
        logging.info("Scanning stopped.")

    def scanning_loop(self):
        """
        Scanning loop that continuously captures frames from the camera.
        """
        fragment_count = 0
        last_time = time.time()
        self.frame_index = 0  # Initialize frame index
        while not self.stop_event.is_set():
            pcd_frame, rgbd = self.capture_point_cloud()

            if pcd_frame and rgbd:
                logging.debug(f"Captured frame {self.frame_index}.")
                self.frame_queue.put((pcd_frame, rgbd))
                fragment_count += 1
            else:
                logging.debug("No valid frame captured, skipping.")

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

        self.total_frames = self.frame_index  # Save total number of frames

    def update_visualization(self, pcd):
        """
        Updates the real-time visualization with the latest point cloud fragment.

        :param pcd: The latest point cloud fragment to visualize.
        """
        if self.vis is None:
            self.initialize_visualization()

        if self.vis is not None:
            pcd_down = pcd.voxel_down_sample(self.voxel_size * 2)
            with threading.Lock():
                self.vis.add_geometry(pcd_down)
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

            # Load saved RGBD frames from disk
            self.load_rgbd_frames()

            # Integrate frames into TSDF volume
            self.integrate_saved_frames()

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
            end_time = datetime.now()
            duration = end_time - start_time
            logging.info(f"3D Scanning ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Total duration: {duration}")

    def load_rgbd_frames(self):
        """
        Loads RGBD frames from saved color and depth images.
        """
        self.rgbd_frames = []
        for frame_index in range(self.frame_index):
            color_filename = os.path.join(self.output_dir, f"color_{frame_index:05d}.png")
            depth_filename = os.path.join(self.output_dir, f"depth_{frame_index:05d}.png")

            if not os.path.exists(color_filename) or not os.path.exists(depth_filename):
                logging.warning(f"Missing files for frame {frame_index}, skipping.")
                continue

            color_image_o3d = o3d.io.read_image(color_filename)
            depth_image_o3d = o3d.io.read_image(depth_filename)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image_o3d,
                depth_image_o3d,
                depth_scale=1.0 / self.depth_scale,
                convert_rgb_to_intensity=False)

            self.rgbd_frames.append(rgbd)
        logging.info(f"Loaded {len(self.rgbd_frames)} RGBD frames from disk.")

    def integrate_saved_frames(self):
        """
        Integrates saved RGBD frames into the TSDF volume.
        """
        logging.info("Integrating saved RGBD frames into TSDF volume...")
        odometry = np.identity(4)
        prev_rgbd = None

        for idx, rgbd in enumerate(self.rgbd_frames):
            if rgbd is not None:
                if prev_rgbd is not None:
                    # Estimate odometry between frames
                    option = o3d.pipelines.odometry.OdometryOption()
                    [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
                        prev_rgbd, rgbd, self.intrinsic, np.identity(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

                    if success:
                        odometry = np.dot(odometry, np.linalg.inv(trans))
                    else:
                        logging.warning(f"Odometry estimation failed for frame {idx}, using previous pose.")

                # Integrate into TSDF volume
                self.tsdf_volume.integrate(rgbd, self.intrinsic, np.linalg.inv(odometry))

                prev_rgbd = rgbd
            else:
                logging.warning(f"No valid RGBD data for frame {idx}, skipping.")

        logging.info("Integration of saved frames complete.")

def parse_arguments():
    """
    Parses command-line arguments for configuration flexibility.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RealSense 3D Scanner with Open3D")
    parser.add_argument("--voxel_size", type=float, default=0.001,
                        help="Voxel size for downsampling (default: 0.001)")
    parser.add_argument("--downsample_voxel_size", type=float, default=0.001,
                        help="Voxel size for final downsampling (default: 0.001)")
    parser.add_argument("--sdf_trunc", type=float, default=0.005,
                        help="Truncation distance for TSDF (default: 0.005)")
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

