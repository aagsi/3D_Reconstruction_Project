import open3d as o3d
import gc

class Visualizer:
    def __init__(self):
        """
        Initializes the visualizer for displaying point clouds.
        """
        self.vis = o3d.visualization.Visualizer()

    def initialize_visualizer(self):
        """
        Creates a window for visualizing 3D point clouds.
        """
        self.vis.create_window(window_name="3D Scan Progress", width=800, height=600)

    def update_visualizer(self, point_cloud):
        """
        Updates the visualizer with the new point cloud.

        :param point_cloud: The point cloud to display.
        """
        self.vis.update_geometry(point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def destroy_visualizer(self):
        """
        Closes the visualizer window.
        """
        self.vis.destroy_window()

    def scanning_loop(self, stop_event, pipeline_manager, point_cloud_capture, point_cloud_alignment, combined_pcd):
        """
        Scanning loop to capture point clouds and update the visualizer in real-time.
        This method runs in a separate thread.

        :param stop_event: A threading event that signals when to stop the scanning loop.
        :param pipeline_manager: The RealSensePipeline object to capture point clouds.
        :param point_cloud_capture: The PointCloudCapture object used to capture point clouds.
        :param point_cloud_alignment: The PointCloudAlignment object used to align point clouds.
        :param combined_pcd: The combined point cloud that will accumulate all frames.
        """
        while not stop_event.is_set():
            # Capture point cloud from the camera
            pcd_frame = point_cloud_capture.capture_point_cloud(pipeline_manager.pipeline)

            if pcd_frame and len(pcd_frame.points) > 0:
                print(f"Captured point cloud with {len(pcd_frame.points)} points.")

                if len(combined_pcd.points) == 0:
                    combined_pcd.points = pcd_frame.points
                    combined_pcd.colors = pcd_frame.colors
                    self.vis.add_geometry(combined_pcd)
                else:
                    # Align the captured frame with the accumulated point cloud
                    pcd_frame_aligned = point_cloud_alignment.align_point_clouds(pcd_frame, combined_pcd)
                    combined_pcd += pcd_frame_aligned

                # Update the visualizer with the new combined point cloud
                self.update_visualizer(combined_pcd)
            else:
                print("No valid point cloud captured, skipping frame.")

            # Free memory
            gc.collect()

