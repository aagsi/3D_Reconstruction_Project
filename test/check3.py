import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import gc
import threading
import cv2
import matplotlib.pyplot as plt
import time


class RealSense3DScanner:
    def __init__(self, voxel_size=0.02, downsample_voxel_size=0.005, cuda_device="CPU:0"):
        """
        Initializes the RealSense3DScanner object with default or user-defined parameters.

        :param voxel_size: The voxel size used for downsampling the point cloud during scanning.
        :param downsample_voxel_size: The voxel size used for final downsampling of the saved point cloud.
        :param cuda_device: The device to be used for operations (e.g., "CUDA:0" or "CPU:0").
        """
        self.pipeline = None
        self.pc = rs.pointcloud()
        self.voxel_size = voxel_size
        self.downsample_voxel_size = downsample_voxel_size
        self.device = o3d.core.Device(cuda_device)
        self.fragments = []  # List to store individual fragments (point clouds)
        self.rgbd_frames = []  # List to store RGBD frames (for TSDF integration)
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

        # Initialize TSDF Volume
        self.tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        # Placeholder for camera intrinsics
        self.intrinsic = None

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Increased FPS for smoother scanning
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            profile = self.pipeline.start(config)
            # Retrieve camera intrinsics
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
            print("Camera intrinsics retrieved successfully.")
        except RuntimeError as e:
            print(f"Failed to start pipeline: {e}")
            device = self.pipeline.get_active_profile().get_device()
            device.hardware_reset()
            self.pipeline.stop()
            exit(1)

    def capture_rgbd_frame(self):
        """
        Captures and returns an RGBD frame from the RealSense camera.

        :return: A tuple (color_image, depth_image) as NumPy arrays, or (None, None) if frames are unavailable.
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap to depth image for visualization (optional)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('Depth', depth_colormap)
        # cv2.imshow('Color', color_image)
        # cv2.waitKey(1)

        return color_image, depth_image

    def capture_point_cloud(self):
        """
        Captures and processes a point cloud from the RealSense camera.

        :return: The processed point cloud, or None if no frames are available.
        """
        color_image, depth_image = self.capture_rgbd_frame()

        if color_image is None or depth_image is None:
            return None, None

        # Store RGBD frames for TSDF integration
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            convert_rgb_to_intensity=False)
        self.rgbd_frames.append(rgbd)

        # Generate point cloud
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image),
            convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d,
            self.intrinsic)

        pcd.transform([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]])  # Align to Open3D coordinate system

        # Downsample the point cloud
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=20))

        return pcd_down, rgbd

    def preprocess_point_cloud(self, pcd):
        """
        Preprocesses the point cloud by estimating normals and computing FPFH features.

        :param pcd: The point cloud to preprocess.
        :return: A tuple of the downsampled point cloud and its FPFH features.
        """
        print("Preprocessing point cloud...")
        # Adjust radii and max_nn for speed
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))

        radius_feature = self.voxel_size * 5
        start_time = time.time()
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=50))
        print(f"Computed FPFH features in {time.time() - start_time:.2f} seconds")

        return pcd, pcd_fpfh

    def register_fragments(self):
        """
        Registers all captured fragments using feature-based registration and multiscale ICP.
        Constructs and optimizes a pose graph for the entire sequence.
        """
        n_fragments = len(self.fragments)
        if n_fragments == 0:
            print("No fragments to register.")
            return

        odometry = np.identity(4)
        self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry.copy()))

        for i in range(1, n_fragments):
            source = self.fragments[i - 1]
            target = self.fragments[i]

            print(f"\nRegistering fragment {i - 1} to fragment {i}")

            source_down, source_fpfh = self.preprocess_point_cloud(source)
            target_down, target_fpfh = self.preprocess_point_cloud(target)

            # Initial alignment using RANSAC
            distance_threshold = self.voxel_size * 1.5
            print("Running RANSAC alignment...")
            start_time = time.time()
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                3,  # Reduced from 4 for speed
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999))
            print(f"RANSAC alignment took {time.time() - start_time:.2f} seconds")

            # Multiscale ICP
            max_correspondence_distances = [self.voxel_size * 15, self.voxel_size * 5, self.voxel_size * 1.5]
            icp_iterations = [30, 20, 10]  # Reduced iterations for speed
            current_transformation = result_ransac.transformation

            for scale in range(len(max_correspondence_distances)):
                distance = max_correspondence_distances[scale]
                print(f"ICP at scale {scale}, max correspondence distance: {distance}")
                start_time = time.time()
                icp_result = o3d.pipelines.registration.registration_icp(
                    source_down, target_down, distance, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_iterations[scale]))
                print(f"ICP at scale {scale} took {time.time() - start_time:.2f} seconds")
                current_transformation = icp_result.transformation

            odometry = np.dot(current_transformation, odometry)
            self.pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
            information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_down, target_down, distance_threshold, current_transformation)
            self.pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i - 1, i, current_transformation, information_matrix, uncertain=False))

        # Pose Graph Optimization
        print("Optimizing pose graph...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.5,
            edge_prune_threshold=0.25,
            preference_loop_closure=0.1,
            reference_node=0)

        o3d.pipelines.registration.global_optimization(
            self.pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        print("Pose graph optimization complete.")

    def integrate_fragments_into_tsdf(self):
        """
        Integrates all RGBD frames into the TSDF volume using the optimized poses.
        """
        print("Integrating RGBD frames into TSDF volume...")
        if len(self.rgbd_frames) != len(self.pose_graph.nodes):
            print("Mismatch between RGBD frames and pose graph nodes.")
            return

        for i, (rgbd, node) in enumerate(zip(self.rgbd_frames, self.pose_graph.nodes)):
            print(f"Integrating frame {i}")
            extrinsic = node.pose
            self.tsdf_volume.integrate(rgbd, self.intrinsic, extrinsic)
        print("TSDF integration complete.")

    def extract_mesh_from_tsdf(self):
        """
        Extracts a mesh from the integrated TSDF volume.
        """
        print("Extracting mesh from TSDF volume...")
        mesh = self.tsdf_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        print("Mesh extraction complete.")
        return mesh

    def start_scanning(self):
        """
        Starts the 3D scanning process in a separate thread.
        """
        stop_event = threading.Event()
        scan_thread = threading.Thread(target=self.scanning_loop, args=(stop_event,))
        scan_thread.start()
        print("Scanning started. Press Enter to stop...")
        input()
        stop_event.set()
        scan_thread.join()
        self.pipeline.stop()
        print("Scanning stopped.")

    def scanning_loop(self, stop_event):
        """
        Scanning loop that continuously captures point clouds from the camera.
        """
        fragment_count = 0
        while not stop_event.is_set():
            pcd_frame, rgbd = self.capture_point_cloud()

            if pcd_frame and len(pcd_frame.points) > 0:
                print(f"Captured fragment {fragment_count} with {len(pcd_frame.points)} points.")
                self.fragments.append(pcd_frame)
                fragment_count += 1
            else:
                print("No valid point cloud captured, skipping frame.")
            gc.collect()

    def save_mesh(self, mesh, filename="output_mesh.ply", color_filename="colored_output_mesh.ply"):
        """
        Saves the reconstructed mesh to files.

        :param mesh: The mesh to save.
        :param filename: Filename for the mesh without colors.
        :param color_filename: Filename for the mesh with color information.
        """
        o3d.io.write_triangle_mesh(filename, mesh)
        print(f"Saved mesh to {filename}")

        # Optionally, color the mesh based on vertex normals or other criteria
        # Here, we color based on vertex normals for visualization
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
        o3d.io.write_triangle_mesh(color_filename, mesh)
        print(f"Saved colored mesh to {color_filename}")

    def run(self):
        """
        Main function that orchestrates the 3D scanning, processing, and saving pipeline.
        """
        self.start_pipeline()
        self.start_scanning()
        self.register_fragments()
        self.integrate_fragments_into_tsdf()
        mesh = self.extract_mesh_from_tsdf()
        self.save_mesh(mesh)


if __name__ == "__main__":
    scanner = RealSense3DScanner()
    scanner.run()

