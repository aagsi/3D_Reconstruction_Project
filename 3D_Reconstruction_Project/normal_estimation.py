import open3d as o3d

class NormalEstimation:
    def __init__(self, device="CUDA:0"):
        """
        Initializes the normal estimation class with the specified CUDA device.

        :param device: The CUDA device to use.
        """
        self.device = o3d.core.Device(device)

    def estimate_normals(self, pcd):
        """
        Estimates normals for the point cloud using CUDA.

        :param pcd: The input point cloud.
        :return: The point cloud with estimated normals.
        """
        pcd_cuda = o3d.t.geometry.PointCloud.from_legacy(pcd, o3d.core.Dtype.Float32, self.device)
        pcd_cuda.estimate_normals(max_nn=50, radius=0.05)
        pcd_cuda.orient_normals_consistent_tangent_plane(100)
        return pcd_cuda.to_legacy()

