import pyrealsense2.pyrealsense2 as rs
import numpy as np
import open3d as o3d
import gc
import threading
import cv2
import matplotlib.pyplot as plt


class RealSense3DScanner:
    def __init__(self, voxel_size=0.01, downsample_voxel_size=0.0025, cuda_device="CUDA:0"):
        """
        Initializes the RealSense3DScanner object with default or user-defined parameters.

        :param voxel_size: The voxel size used for downsampling the point cloud during real-time scanning.
        :param downsample_voxel_size: The voxel size used for final downsampling of the saved point cloud.
        :param cuda_device: The CUDA device to be used for GPU operations (e.g., "CUDA:0").
        """
        self.pipeline = None
        self.pc = rs.pointcloud()
        self.combined_pcd = o3d.geometry.PointCloud()
        self.voxel_size = voxel_size
        self.downsample_voxel_size = downsample_voxel_size
        self.device = o3d.core.Device(cuda_device)
        self.vis = o3d.visualization.Visualizer()

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        Configures the streams and handles hardware reset in case of initialization failure.
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
        Captures a point cloud from the RealSense camera by combining the depth and color streams.
        Converts the captured point cloud to a tensor-based point cloud for CUDA operations and downsamples it.

        :return: The downsampled point cloud in legacy format, or None if no frames are available.
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
        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(vtx)
        pcd_frame.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

        pcd_frame_cuda = o3d.t.geometry.PointCloud.from_legacy(pcd_frame, o3d.core.Dtype.Float32, self.device)
        pcd_frame_cuda = pcd_frame_cuda.voxel_down_sample(self.voxel_size)

        return pcd_frame_cuda.to_legacy()

    def align_point_clouds(self, source, target, threshold=0.02):
        """
        Aligns two point clouds using the Iterative Closest Point (ICP) algorithm. The source point cloud is aligned
        to the target point cloud based on minimizing the distance between corresponding points.

        :param source: The source point cloud to be aligned.
        :param target: The target point cloud to which the source will be aligned.
        :param threshold: Maximum allowed distance between corresponding points for alignment.
        :return: The aligned source point cloud.
        """
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        source.transform(reg_p2p.transformation)
        return source

    def initialize_visualizer(self):
        """
        Initializes an Open3D visualizer window to display real-time progress of the 3D scan.
        The visualizer is used to show the combined point clouds as they are captured.
        """
        self.vis.create_window(window_name="3D Scan Progress", width=800, height=600)

    def start_scanning(self):
        """
        Starts the 3D scanning process in a separate thread, continuously capturing and aligning point clouds.
        Allows the user to stop the scanning process by pressing Enter.
        """
        stop_event = threading.Event()
        scan_thread = threading.Thread(target=self.scanning_loop, args=(stop_event,))
        scan_thread.start()
        input("Press Enter to stop scanning...\n")
        stop_event.set()
        scan_thread.join()
        self.vis.destroy_window()
        self.pipeline.stop()

    def scanning_loop(self, stop_event):
        """
        Scanning loop that continuously captures point clouds from the camera, aligns them to the combined
        point cloud, and updates the visualizer in real-time. This function runs in a separate thread.

        :param stop_event: A threading event that signals when to stop the scanning loop.
        """
        while not stop_event.is_set():
            pcd_frame = self.capture_point_cloud()

            if pcd_frame and len(pcd_frame.points) > 0:
                print(f"Captured point cloud with {len(pcd_frame.points)} points.")
                if len(self.combined_pcd.points) == 0:
                    self.combined_pcd.points = pcd_frame.points
                    self.combined_pcd.colors = pcd_frame.colors
                    self.vis.add_geometry(self.combined_pcd)
                else:
                    pcd_frame_aligned = self.align_point_clouds(pcd_frame, self.combined_pcd)
                    self.combined_pcd += pcd_frame_aligned

                self.vis.update_geometry(self.combined_pcd)
                self.vis.poll_events()
                self.vis.update_renderer()
            else:
                print("No valid point cloud captured, skipping frame.")
            gc.collect()

    def save_point_cloud(self, filename="captured_data_on_the_fly.ply"):
        """
        Saves the combined point cloud to a file in PLY format after scanning is complete.

        :param filename: The name of the output file to save the point cloud.
        """
        if len(self.combined_pcd.points) > 0:
            o3d.io.write_point_cloud(filename, self.combined_pcd)
            print(f"Saved point cloud to {filename}")
        else:
            print("No valid frames captured.")
            raise RuntimeError("No valid frames captured.")

    def process_point_cloud(self, filename="captured_data_on_the_fly.ply"):
        """
        Loads the saved point cloud from a file and processes it by removing noise and downsampling the points.

        :param filename: The name of the file containing the saved point cloud.
        :return: The processed point cloud with noise removed and downsampled.
        """
        pcd = o3d.io.read_point_cloud(filename)

        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd, o3d.core.Dtype.Float32, self.device)
        pcd_down = pcd_t.voxel_down_sample(self.downsample_voxel_size)

        pcd_down_legacy = pcd_down.to_legacy()

        cl, ind = pcd_down_legacy.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
        pcd_inlier = pcd_down_legacy.select_by_index(ind)

        pcd_inlier, ind = pcd_inlier.remove_radius_outlier(nb_points=16, radius=0.01)

        return pcd_inlier

    def estimate_normals(self, pcd):
        """
        Estimates normals for the point cloud using CUDA and ensures consistent orientation of the normals.

        :param pcd: The point cloud for which normals will be estimated.
        :return: The point cloud with estimated normals.
        """
        pcd_cuda = o3d.t.geometry.PointCloud.from_legacy(pcd, o3d.core.Dtype.Float32, self.device)
        pcd_cuda.estimate_normals(max_nn=50, radius=0.05)
        pcd_cuda.orient_normals_consistent_tangent_plane(100)
        return pcd_cuda.to_legacy()

    def reconstruct_mesh(self, pcd_inlier, depth=6):
        """
        Reconstructs a 3D mesh from the processed point cloud using Poisson surface reconstruction.

        :param pcd_inlier: The processed point cloud from which the mesh will be created.
        :param depth: The depth of the Poisson surface reconstruction (higher depth gives more detail).
        :return: The reconstructed mesh and corresponding density values.
        """
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_inlier, depth=depth)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
        return mesh, densities

    def save_mesh(self, mesh, densities, filename="output_mesh_on_the_fly.ply", color_filename="colored_output_mesh_on_the_fly.ply"):
        """
        Saves the reconstructed mesh to a file in PLY format, and optionally saves a density-colored version.

        :param mesh: The reconstructed mesh to be saved.
        :param densities: The density values used for coloring the mesh.
        :param filename: The filename for saving the standard mesh.
        :param color_filename: The filename for saving the density-colored mesh.
        """
        o3d.io.write_triangle_mesh(filename, mesh)
        densities_np = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')((densities_np - densities_np.min()) / (densities_np.max() - densities_np.min()))
        mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors[:, :3])
        o3d.io.write_triangle_mesh(color_filename, mesh)

    def run(self):
        """
        The main function that orchestrates the 3D scanning, processing, and saving pipeline.
        It starts the RealSense camera, initiates the visualizer, runs the scanning loop, processes the captured point cloud,
        estimates normals, reconstructs a mesh, and saves the results to PLY files.
        """
        self.start_pipeline()
        self.initialize_visualizer()
        self.start_scanning()
        self.save_point_cloud()
        processed_pcd = self.process_point_cloud()
        pcd_with_normals = self.estimate_normals(processed_pcd)
        mesh, densities = self.reconstruct_mesh(pcd_with_normals)
        self.save_mesh(mesh, densities)


if __name__ == "__main__":
    scanner = RealSense3DScanner()
    scanner.run()

