import pyrealsense2.pyrealsense2 as rs
import numpy as np
import open3d as o3d
import gc
import threading
import cv2

# Adjustable Parameters
VOXEL_SIZE = 0.05  # Voxel size for downsampling during scanning
DOWNSAMPLE_VOXEL_SIZE = 0.01  # Voxel size for final downsampling
CUDA_DEVICE = "CUDA:0"  # CUDA device for GPU operations (if applicable)
DEPTH_FILTERS = True  # Apply depth filters to reduce noise
ALIGNMENT_THRESHOLD = 0.1  # Threshold for alignment
MAX_ITERATION = 50  # Max iterations for alignment
INTEGRATE_TSDF = True  # Use TSDF integration for reconstruction
VOXEL_LENGTH = 0.04  # Voxel length for TSDF volume
SDF_TRUNC = 0.1  # Truncation value for TSDF volume
CAMERA_INTRINSIC = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
)  # Camera intrinsic parameters
DEPTH_SCALE = 1000.0  # Scale factor for depth values (RealSense default is 1000)
DEPTH_TRUNC = 3.0  # Truncate depth beyond this value
DEPTH_MIN = 0.1  # Minimum depth value
USE_VISUALIZATION = False  # Set to True to enable visualization

class RealSense3DScanner:
    def __init__(self):
        """
        Initializes the RealSense3DScanner object with adjustable parameters.
        """
        self.pipeline = None
        self.pc = rs.pointcloud()
        self.voxel_size = VOXEL_SIZE
        self.downsample_voxel_size = DOWNSAMPLE_VOXEL_SIZE
        self.device = o3d.core.Device(CUDA_DEVICE)
        self.previous_rgbd = None
        self.global_transformation = np.eye(4)
        self.volume = None
        if INTEGRATE_TSDF:
            self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=VOXEL_LENGTH,
                sdf_trunc=SDF_TRUNC,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
        self.intrinsic = CAMERA_INTRINSIC
        self.depth_scale = DEPTH_SCALE
        self.depth_trunc = DEPTH_TRUNC
        self.depth_min = DEPTH_MIN
        self.use_visualization = USE_VISUALIZATION
        if self.use_visualization:
            self.vis = o3d.visualization.Visualizer()
            self.vis_created = False
            self.pcd_vis = o3d.geometry.PointCloud()

        # Initialize align object to align depth to color frame
        self.align = rs.align(rs.stream.color)

    def start_pipeline(self):
        """
        Starts the RealSense pipeline for capturing color and depth streams.
        Configures the streams and handles hardware reset in case of initialization failure.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            print(f"Failed to start pipeline: {e}")
            device = self.pipeline.get_active_profile().get_device()
            device.hardware_reset()
            self.pipeline.stop()
            exit(1)

    def capture_rgbd_image(self):
        """
        Captures an RGB-D image from the RealSense camera with optional filters applied.

        :return: The captured RGB-D image.
        """
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        # Optionally apply filters to the depth frame
        if DEPTH_FILTERS:
            # Apply filters
            spatial = rs.spatial_filter()
            temporal = rs.temporal_filter()
            hole_filling = rs.hole_filling_filter()

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Ensure that depth and color images are the same size
        if depth_image.shape[:2] != color_image.shape[:2]:
            print("Depth and color image sizes do not match. Resizing color image.")
            color_image = cv2.resize(color_image, (depth_image.shape[1], depth_image.shape[0]))

        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )

        return rgbd_image

    def align_rgbd_images(self, source_rgbd, target_rgbd, prev_transformation):
        """
        Aligns two RGB-D frames using RGB-D odometry.

        :param source_rgbd: The source RGB-D image.
        :param target_rgbd: The target RGB-D image.
        :param prev_transformation: The previous transformation matrix.
        :return: The transformation matrix from source to target.
        """
        option = o3d.pipelines.odometry.OdometryOption()
        # Commented out the line causing the error
        # option.max_depth_diff = 0.07  # Adjust as needed

        odo_init = prev_transformation

        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd, target_rgbd, self.intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

        if success:
            return trans
        else:
            return None

    def start_scanning(self):
        """
        Starts the 3D scanning process in a separate thread, continuously capturing and aligning RGB-D images.
        Allows the user to stop the scanning process by pressing Enter.
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
        Scanning loop that uses RGB-D odometry for alignment and integrates frames into TSDF volume.
        """
        prev_transformation = np.eye(4)

        while not stop_event.is_set():
            rgbd_image = self.capture_rgbd_image()

            if rgbd_image is not None:
                print("Captured RGB-D image.")
                if self.previous_rgbd is None:
                    self.previous_rgbd = rgbd_image
                    if INTEGRATE_TSDF:
                        self.volume.integrate(rgbd_image, self.intrinsic, np.linalg.inv(prev_transformation))
                else:
                    trans = self.align_rgbd_images(rgbd_image, self.previous_rgbd, prev_transformation)
                    if trans is not None:
                        prev_transformation = np.dot(trans, prev_transformation)
                        if INTEGRATE_TSDF:
                            self.volume.integrate(rgbd_image, self.intrinsic, np.linalg.inv(prev_transformation))
                        self.previous_rgbd = rgbd_image
                    else:
                        print("Odometry failed. Skipping frame.")
                if self.use_visualization:
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image, self.intrinsic)
                    pcd.transform(prev_transformation)
                    self.update_visualization(pcd)
            else:
                print("No valid frames captured, skipping.")
            gc.collect()

    def update_visualization(self, pcd):
        """
        Updates the visualization with the latest point cloud.

        :param pcd: The point cloud to display.
        """
        if not self.vis_created:
            self.vis_created = True
            self.vis.create_window("Real-Time Scanning")
            self.pcd_vis = pcd
            self.vis.add_geometry(self.pcd_vis)
        else:
            self.pcd_vis.points = pcd.points
            self.pcd_vis.colors = pcd.colors
            self.vis.update_geometry(self.pcd_vis)
            self.vis.poll_events()
            self.vis.update_renderer()

    def save_mesh(self, filename="output_mesh_on_the_fly.ply"):
        """
        Extracts the mesh from TSDF volume and saves it to a file.

        :param filename: The filename for saving the mesh.
        """
        if INTEGRATE_TSDF:
            print("Extracting mesh from TSDF volume...")
            mesh = self.volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            if len(mesh.vertices) > 0:
                o3d.io.write_triangle_mesh(filename, mesh)
                print(f"Saved mesh to {filename}")
            else:
                print("Mesh extraction resulted in an empty mesh.")
        else:
            print("TSDF integration is disabled. Cannot extract mesh.")

    def run(self):
        """
        The main function that orchestrates the 3D scanning and reconstruction pipeline.
        It starts the RealSense camera, runs the scanning loop, and saves the reconstructed mesh.
        """
        self.start_pipeline()
        self.start_scanning()
        if self.use_visualization:
            self.vis.destroy_window()
        self.save_mesh()


if __name__ == "__main__":
    scanner = RealSense3DScanner()
    scanner.run()

