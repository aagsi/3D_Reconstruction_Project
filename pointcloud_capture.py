import numpy as np
import open3d as o3d
import pyrealsense2.pyrealsense2 as rs

class PointCloudCapture:
    def __init__(self, device="CUDA:0", voxel_size=0.01):
        """
        Initializes the class for capturing point clouds.

        :param device: The CUDA device to use.
        :param voxel_size: The voxel size for downsampling.
        """
        self.pc = rs.pointcloud()
        self.device = o3d.core.Device(device)
        self.voxel_size = voxel_size

    def capture_point_cloud(self, pipeline):
        """
        Captures a point cloud from the RealSense camera.

        :param pipeline: The RealSense pipeline object.
        :return: The captured and downsampled point cloud.
        """
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Capture point cloud from depth frame
        points = self.pc.calculate(depth_frame)
        self.pc.map_to(color_frame)

        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        color_image = color_image.reshape(-1, 3) / 255.0

        # Create Open3D point cloud
        pcd_frame = o3d.geometry.PointCloud()
        pcd_frame.points = o3d.utility.Vector3dVector(vtx)
        pcd_frame.colors = o3d.utility.Vector3dVector(color_image)

        # Convert to CUDA-enabled point cloud
        pcd_frame_cuda = o3d.t.geometry.PointCloud.from_legacy(pcd_frame, o3d.core.Dtype.Float32, self.device)

        # Downsample using CUDA
        pcd_frame_cuda = pcd_frame_cuda.voxel_down_sample(self.voxel_size)

        # Convert back to legacy format
        pcd_frame_legacy = pcd_frame_cuda.to_legacy()

        return pcd_frame_legacy

