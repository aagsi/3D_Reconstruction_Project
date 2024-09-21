import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import gc
import threading
import cv2
import matplotlib.pyplot as plt
import time
import argparse
import sys
import os
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor


class RealSense3DScanner:
    def __init__(self, voxel_size=0.02, downsample_voxel_size=0.005,
                 sdf_trunc=0.04, max_fps=30, cuda_device="CPU:0",
                 output_dir="output", visualize=False):
        """
        Initializes the RealSense3DScanner object with default or user-defined parameters.

        :param voxel_size: The voxel size used for downsampling the point cloud during scanning.
        :param downsample_voxel_size: The voxel size used for final downsampling of the saved point cloud.
        :param sdf_trunc: The truncation distance for TSDF integration.
        :param max_fps: Maximum frames per second for RealSense streams.
        :param cuda_device: The device to be used for operations (e.g., "CUDA:0" or "CPU:0").
        :param output_dir: Directory where output files will be saved.
        :param visualize: Boolean indicating whether to visualize the scanning process in real-time.
        """
        # Initialize parameters
        self.voxel_size = voxel_size
        self.downsample_voxel_size = downsample_voxel_size
        self.sdf_trunc = sdf_trunc
        self.max_fps = max_fps
        self.device = o3d.core.Device(cuda_device)
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

    def is_registration_good(self, fitness, rmse):
        """
        Determines if the registration result meets quality thresholds.

        :param fitness: Fitness score of the registration.
        :param rmse: RMSE of the registration.
        :return: Boolean indicating if registration is good.
        """
        FITNESS_THRESHOLD = 0.3
        RMSE_THRESHOLD = 0.02
        return fitness >= FITNESS_THRESHOLD and rmse <= RMSE_THRESHOLD

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        """
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, self.max_fps)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.max_fps)

            profile = self.pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
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
            if self.pipeline:
                device = self.pipeline.get_active_profile().get_device()
                device.hardware_reset()
                self.pipeline.stop()
            sys.exit(1)

    def capture_rgbd_frame(self, max_retries=3, retry_delay=0.1, timeout_ms=500):
        """
        Captures and returns an RGBD frame from the RealSense camera with retry mechanism.

        :param max_retries: Maximum number of retries for frame capture.
        :param retry_delay: Delay between retries in seconds.
        :param timeout_ms: Timeout for frame capture in milliseconds.
        :return: A tuple (color_image, depth_image) as NumPy arrays, or (None, None) if frames are unavailable.
        """
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                elapsed_time = time.time() - start_time
                logging.info(f"Frame capture time: {elapsed_time:.3f} seconds")
                if not frames:
                    logging.warning("No frames captured.")
                    raise RuntimeError("No frames captured.")
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    logging.warning("Incomplete frames received.")
                    raise RuntimeError("Incomplete frames received.")

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                return color_image, depth_image
            except RuntimeError as e:
                logging.error(f"Attempt {attempt + 1}/{max_retries} - Failed to capture frames: {e}")
                time.sleep(retry_delay)  # Wait before retrying
        logging.warning("All frame capture attempts failed.")
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
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_image),
                o3d.geometry.Image(depth_image),
                convert_rgb_to_intensity=False)
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

    def preprocess_point_cloud_threaded(self, pcd):
        """
        Preprocesses the point cloud in a separate thread.

        :param pcd: The point cloud to preprocess.
        :return: A tuple of the downsampled point cloud and its FPFH features.
        """
        return self.preprocess_point_cloud(pcd)

    def preprocess_point_cloud(self, pcd):
        """
        Preprocesses the point cloud by estimating normals and computing FPFH features.

        :param pcd: The point cloud to preprocess.
        :return: A tuple of the downsampled point cloud and its FPFH features.
        """
        try:
            logging.info("Preprocessing point cloud...")
            radius_normal = self.voxel_size * 2
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))

            radius_feature = self.voxel_size * 5
            start_time = time.time()
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))
            logging.info(f"Computed FPFH features in {time.time() - start_time:.2f} seconds")

            return pcd, pcd_fpfh
        except Exception as e:
            logging.error(f"Preprocessing failed: {e}")
            return None, None

    def register_fragments(self):
        """
        Registers all captured fragments using feature-based registration and multiscale ICP.
        Constructs and optimizes a pose graph for the entire sequence.
        """
        n_fragments = len(self.fragments)
        if n_fragments == 0:
            logging.warning("No fragments to register.")
            return

        odometry = np.identity(4)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry.copy()))

        for i in range(1, n_fragments):
            source = self.fragments[i - 1]
            target = self.fragments[i]

            logging.info(f"Registering fragment {i - 1} to fragment {i}")

            source_down, source_fpfh = self.preprocess_point_cloud(source)
            target_down, target_fpfh = self.preprocess_point_cloud(target)

            if source_down is None or target_down is None:
                logging.warning(f"Skipping registration between fragment {i - 1} and {i} due to preprocessing failure.")
                continue

            distance_threshold = self.voxel_size * 1.5
            try:
                result_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                    source_down, target_down, source_fpfh, target_fpfh,
                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                        maximum_correspondence_distance=distance_threshold))
                current_transformation = result_fgr.transformation
                logging.info("Fast Global Registration successful.")
            except Exception as e:
                logging.error(f"Fast Global Registration failed: {e}")
                continue

            # Multiscale ICP
            max_correspondence_distances = [self.voxel_size * 15, self.voxel_size * 5, self.voxel_size * 1.5]
            icp_iterations = [30, 20, 10]
            for scale in range(len(max_correspondence_distances)):
                distance = max_correspondence_distances[scale]
                logging.info(f"ICP at scale {scale}, max correspondence distance: {distance}")
                try:
                    icp_result = o3d.pipelines.registration.registration_icp(
                        source_down, target_down, distance, current_transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iterations[scale]))
                    logging.info(f"ICP at scale {scale} completed with fitness {icp_result.fitness:.4f} and RMSE {icp_result.inlier_rmse:.4f}")
                    current_transformation = icp_result.transformation

                    # Check registration quality
                    if not self.is_registration_good(icp_result.fitness, icp_result.inlier_rmse):
                        logging.warning(f"ICP at scale {scale} failed quality check. Skipping fragment {i}.")
                        raise ValueError("Poor registration quality.")
                except Exception as e:
                    logging.error(f"ICP at scale {scale} failed: {e}")
                    break  # Exit ICP scales on failure

            # Update odometry
            odometry = np.dot(current_transformation, odometry)

            if not np.all(np.isfinite(odometry)):
                logging.warning(f"Invalid odometry transformation for fragment {i}. Skipping pose graph node addition.")
                continue

            self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))

            try:
                information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    source_down, target_down, distance_threshold, current_transformation)
                self.pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(i - 1, i, current_transformation, information_matrix, uncertain=False))
            except Exception as e:
                logging.error(f"Failed to create pose graph edge between {i - 1} and {i}: {e}")
                continue

        # Pose Graph Optimization
        logging.info("Optimizing pose graph...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.5,
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
            logging.info(f"Integrating frame {i}")
            extrinsic = node.pose

            if not np.all(np.isfinite(extrinsic)):
                logging.warning(f"Frame {i} has invalid extrinsic matrix. Skipping integration.")
                continue

            try:
                self.tsdf_volume.integrate(rgbd, self.intrinsic, extrinsic)
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
        stop_event = threading.Event()
        scan_thread = threading.Thread(target=self.scanning_loop, args=(stop_event,))
        scan_thread.start()
        logging.info("Scanning started. Press Enter to stop...")
        input()
        stop_event.set()
        scan_thread.join()
        self.pipeline.stop()
        logging.info("Scanning stopped.")

    def scanning_loop(self, stop_event):
        """
        Scanning loop that continuously captures point clouds from the camera.
        """
        fragment_count = 0
        initial_frames = 10  # Allow some initial frames to warm up
        while not stop_event.is_set():
            pcd_frame, rgbd = self.capture_point_cloud()

            if pcd_frame and len(pcd_frame.points) > 0:
                logging.info(f"Captured fragment {fragment_count} with {len(pcd_frame.points)} points.")
                self.fragments.append(pcd_frame)
                fragment_count += 1

                if self.visualize:
                    self.update_visualization(pcd_frame)
            else:
                if fragment_count < initial_frames:
                    logging.info(f"Skipping initial bad frame {fragment_count}.")
                else:
                    logging.warning("No valid point cloud captured, skipping frame.")
            gc.collect()

    def update_visualization(self, pcd):
        """
        Updates the real-time visualization with the latest point cloud fragment.

        :param pcd: The latest point cloud fragment to visualize.
        """
        if self.vis is None:
            self.initialize_visualization()

        if self.vis is not None:
            self.vis.add_geometry(pcd)
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

    def save_mesh(self, mesh, filename="output_mesh.ply", color_filename="colored_output_mesh.ply"):
        """
        Saves the reconstructed mesh to files.

        :param mesh: The mesh to save.
        :param filename: Filename for the mesh without colors.
        :param color_filename: Filename for the mesh with color information.
        """
        try:
            mesh_filepath = os.path.join(self.output_dir, filename)
            o3d.io.write_triangle_mesh(mesh_filepath, mesh)
            logging.info(f"Saved mesh to {mesh_filepath}")

            # Optionally, color the mesh based on vertex normals or other criteria
            mesh.paint_uniform_color([0.5, 0.5, 0.5])
            color_mesh_filepath = os.path.join(self.output_dir, color_filename)
            o3d.io.write_triangle_mesh(color_mesh_filepath, mesh)
            logging.info(f"Saved colored mesh to {color_mesh_filepath}")
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
                color_mesh_filename = f"colored_output_mesh_{timestamp}.ply"
                self.save_mesh(mesh, filename=mesh_filename, color_filename=color_mesh_filename)
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
    parser.add_argument("--cuda_device", type=str, default="CPU:0",
                        help='Device for operations, e.g., "CUDA:0" or "CPU:0" (default: "CPU:0")')
    parser.add_argument("--output_dir", type=str, default="output",
                        help='Directory to save output files (default: "output")')
    parser.add_argument("--visualize", action='store_true',
                        help="Enable real-time visualization (default: False)")
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
        visualize=args.visualize
    )
    scanner.run()

