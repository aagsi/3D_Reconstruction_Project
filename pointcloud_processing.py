import open3d as o3d
import numpy as np

class PointCloudProcessingWithCUDA:
    def __init__(self, device="CUDA:0", downsample_voxel_size=0.0025):
        """
        Initializes the point cloud processing class with GPU support.

        :param device: The CUDA device to use.
        :param downsample_voxel_size: The voxel size for downsampling.
        """
        self.device = o3d.core.Device(device)
        self.downsample_voxel_size = downsample_voxel_size

    def process_point_cloud(self, filename):
        """
        Loads and processes a point cloud file using GPU, including downsampling and noise removal.

        :param filename: The name of the point cloud file to process.
        :return: The processed point cloud.
        """
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(filename)
        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd, o3d.core.Dtype.Float32, self.device)

        # GPU-accelerated voxel downsampling
        pcd_down = pcd_t.voxel_down_sample(self.downsample_voxel_size)

        # Convert to a legacy point cloud for visualization or if needed (if required by another function)
        pcd_down_legacy = pcd_down.to_legacy()

        # GPU-accelerated statistical outlier removal (replace legacy method)
        # For now, Open3D only supports CPU-based statistical outlier removal, so we can keep this part on the CPU.
        # GPU-accelerated statistical removal could be implemented via custom CUDA kernels or future Open3D updates.
        cl, ind = pcd_down_legacy.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
        pcd_inlier_legacy = pcd_down_legacy.select_by_index(ind)

        # Remove radius outliers (also still on CPU)
        pcd_inlier_legacy, ind = pcd_inlier_legacy.remove_radius_outlier(nb_points=16, radius=0.01)

        # Convert back to Tensor-based GPU point cloud for any further GPU processing
        pcd_inlier = o3d.t.geometry.PointCloud.from_legacy(pcd_inlier_legacy, o3d.core.Dtype.Float32, self.device)

        return pcd_inlier_legacy  # Or return pcd_inlier if GPU processing is required in later steps

