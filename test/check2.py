import pyrealsense2.pyrealsense2 as rs
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
        self.fragments = []  # List to store individual fragments
        self.pose_graph = o3d.pipelines.registration.PoseGraph()

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            print(f"Failed to start pipeline: {e}")
            device = self.pipeline.get_active_profile().get_device()
            device.hardware_reset()
            self.pipeline.stop()
            exit(1)

    def capture_point_cloud(self):
        """
        Captures and processes a point cloud from the RealSense camera.

        :return: The processed point cloud, or None if no frames are available.
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        points = self.pc.calculate(depth_frame)
        self.pc.map_to(color_frame)

        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # Map texture coordinates to color image
        u = (tex_coords[:, 0] * color_image.shape[1]).astype(np.int32)
        v = (tex_coords[:, 1] * color_image.shape[0]).astype(np.int32)
        valid_idx = (u >= 0) & (u < color_image.shape[1]) & (v >= 0) & (v < color_image.shape[0])
        u = u[valid_idx]
        v = v[valid_idx]
        vtx = vtx[valid_idx]

        colors = color_image[v, u] / 255.0

        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(vtx)
        pcd_frame.colors = o3d.utility.Vector3dVector(colors)

        # Downsample the point cloud
        pcd_frame_down = pcd_frame.voxel_down_sample(self.voxel_size)
        pcd_frame_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=20))

        return pcd_frame_down

    def preprocess_point_cloud(self, pcd):
        """
        Preprocesses the point cloud by estimating normals and computing FPFH features.

        :param pcd: The point cloud to preprocess.
        :return: A tuple of the point cloud and its FPFH features.
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
        """
        n_fragments = len(self.fragments)
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

    def integrate_fragments(self):
        """
        Integrates the registered fragments into a single point cloud using the optimized poses.
        """
        print("Integrating fragments...")
        combined_pcd = o3d.geometry.PointCloud()
        for i, pcd in enumerate(self.fragments):
            print(f"Integrating fragment {i}")
            pcd_transformed = pcd.transform(self.pose_graph.nodes[i].pose)
            combined_pcd += pcd_transformed
        combined_pcd_down = combined_pcd.voxel_down_sample(self.downsample_voxel_size)
        return combined_pcd_down

    def start_scanning(self):
        """
        Starts the 3D scanning process in a separate thread.
        """
        stop_event = threading.Event()
        scan_thread = threading.Thread(target=self.scanning_loop, args=(stop_event,))
        scan_thread.start()
        input("Press Enter to stop scanning...\n")
        stop_event.set()
        scan_thread.join()
        self.pipeline.stop()

    def scanning_loop(self, stop_event):
        """
        Scanning loop that continuously captures point clouds from the camera.
        """
        fragment_count = 0
        while not stop_event.is_set():
            pcd_frame = self.capture_point_cloud()

            if pcd_frame and len(pcd_frame.points) > 0:
                print(f"Captured fragment {fragment_count} with {len(pcd_frame.points)} points.")
                self.fragments.append(pcd_frame)
                fragment_count += 1
            else:
                print("No valid point cloud captured, skipping frame.")
            gc.collect()

    def save_point_cloud(self, pcd, filename="combined_data.ply"):
        """
        Saves the combined point cloud to a file.
        """
        if len(pcd.points) > 0:
            o3d.io.write_point_cloud(filename, pcd)
            print(f"Saved point cloud to {filename}")
        else:
            print("No valid frames captured.")
            raise RuntimeError("No valid frames captured.")

    def process_point_cloud(self, pcd):
        """
        Processes the point cloud by removing noise and downsampling.
        """
        print("Processing combined point cloud...")
        pcd_down = pcd.voxel_down_sample(self.downsample_voxel_size)

        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_inlier = pcd_down.select_by_index(ind)

        pcd_inlier, ind = pcd_inlier.remove_radius_outlier(nb_points=16, radius=0.02)

        return pcd_inlier

    def estimate_normals(self, pcd):
        """
        Estimates normals for the point cloud.
        """
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)
        return pcd

    def reconstruct_mesh(self, pcd_inlier, depth=8):
        """
        Reconstructs a 3D mesh from the point cloud.
        """
        print("Reconstructing mesh...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_inlier, depth=depth)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
        print("Mesh reconstruction complete.")
        return mesh, densities

    def save_mesh(self, mesh, densities, filename="output_mesh.ply", color_filename="colored_output_mesh.ply"):
        """
        Saves the reconstructed mesh to files.
        """
        o3d.io.write_triangle_mesh(filename, mesh)
        densities_np = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')((densities_np - densities_np.min()) / (densities_np.max() - densities_np.min()))
        mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors[:, :3])
        o3d.io.write_triangle_mesh(color_filename, mesh)
        print(f"Saved meshes to {filename} and {color_filename}")

    def run(self):
        """
        Main function that orchestrates the 3D scanning, processing, and saving pipeline.
        """
        self.start_pipeline()
        self.start_scanning()
        self.register_fragments()
        combined_pcd = self.integrate_fragments()
        self.save_point_cloud(combined_pcd, filename="combined_data.ply")
        processed_pcd = self.process_point_cloud(combined_pcd)
        pcd_with_normals = self.estimate_normals(processed_pcd)
        mesh, densities = self.reconstruct_mesh(pcd_with_normals)
        self.save_mesh(mesh, densities)


if __name__ == "__main__":
    scanner = RealSense3DScanner()
    scanner.run()

