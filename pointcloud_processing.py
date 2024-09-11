import open3d as o3d
import numpy as np

class PointCloudProcessing:
    def __init__(self, device="CUDA:0", downsample_voxel_size=0.0025):
        """
        Initializes the point cloud processing class.

        :param device: The CUDA device to use.
        :param downsample_voxel_size: The voxel size for downsampling.
        """
        self.device = o3d.core.Device(device)
        self.downsample_voxel_size = downsample_voxel_size

    def process_point_cloud(self, filename):
        """
        Loads and processes a point cloud file, including downsampling and noise removal.

        :param filename: The name of the point cloud file to process.
        :return: The processed point cloud.
        """
        pcd = o3d.io.read_point_cloud(filename)
        pcd_t = o3d.t.geometry.PointCloud.from_legacy(pcd, o3d.core.Dtype.Float32, self.device)
        pcd_down = pcd_t.voxel_down_sample(self.downsample_voxel_size)
        pcd_down_legacy = pcd_down.to_legacy()

        cl, ind = pcd_down_legacy.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
        pcd_inlier = pcd_down_legacy.select_by_index(ind)

        pcd_inlier, ind = pcd_inlier.remove_radius_outlier(nb_points=16, radius=0.01)

        return pcd_inlier

