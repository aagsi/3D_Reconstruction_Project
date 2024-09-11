import numpy as np
import open3d as o3d

class PointCloudAlignment:
    def align_point_clouds(self, source, target, threshold=0.02):
        """
        Aligns two point clouds using ICP algorithm.

        :param source: The source point cloud.
        :param target: The target point cloud.
        :param threshold: The ICP threshold for point-to-point registration.
        :return: The aligned source point cloud.
        """
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        source.transform(reg_p2p.transformation)
        return source

