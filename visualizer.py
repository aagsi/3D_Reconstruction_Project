import open3d as o3d
import numpy as np
import gc

class GeometryVisualizer:
    def __init__(self):
        """
        Initializes the visualizer for displaying reconstructed geometry.
        """
        self.vis = o3d.visualization.Visualizer()

    def initialize_visualizer(self):
        """
        Creates a window for visualizing 3D reconstructed geometry.
        """
        self.vis.create_window(window_name="3D Reconstruction Progress", width=800, height=600)

    def update_visualizer(self, mesh):
        """
        Updates the visualizer with the new reconstructed mesh.

        :param mesh: The reconstructed geometry (mesh) to display.
        """
        self.vis.clear_geometries()
        self.vis.add_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()

    def destroy_visualizer(self):
        """
        Closes the visualizer window.
        """
        self.vis.destroy_window()

    def highlight_sparse_regions(self, mesh, densities, threshold=0.01):
        """
        Highlights sparse regions of the mesh with gaps or low-density areas.

        :param mesh: The reconstructed geometry (mesh) to highlight.
        :param densities: The density of vertices in the mesh, used to detect sparse areas.
        :param threshold: Density threshold to consider a region sparse.
        :return: Mesh with highlighted sparse regions.
        """
        # Color the mesh based on vertex density
        colors = np.zeros((np.asarray(mesh.vertices).shape[0], 3))  # Default color: black
        low_density_vertices = np.asarray(densities) < threshold
        colors[low_density_vertices] = [1, 0, 0]  # Mark sparse areas in red

        # Apply the color to the mesh
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        return mesh

    def estimate_normals(self, pcd):
        """
        Estimate normals for the point cloud.
        :param pcd: The point cloud to estimate normals for.
        """
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

    def scanning_loop(self, stop_event, pipeline_manager, point_cloud_capture, point_cloud_alignment, combined_pcd, mesh_reconstruction):
        """
        Scanning loop to capture point clouds, reconstruct geometry, and update the visualizer in real-time.
        This method runs in a separate thread.

        :param stop_event: A threading event that signals when to stop the scanning loop.
        :param pipeline_manager: The RealSensePipeline object to capture point clouds.
        :param point_cloud_capture: The PointCloudCapture object used to capture point clouds.
        :param point_cloud_alignment: The PointCloudAlignment object used to align point clouds.
        :param combined_pcd: The combined point cloud that will accumulate all frames.
        :param mesh_reconstruction: The mesh reconstruction object.
        """
        self.initialize_visualizer()  # Initialize visualizer at the start

        while not stop_event.is_set():
            # Capture point cloud from the camera
            pcd_frame = point_cloud_capture.capture_point_cloud(pipeline_manager.pipeline)

            if pcd_frame and len(pcd_frame.points) > 0:
                print(f"Captured point cloud with {len(pcd_frame.points)} points.")

                if len(combined_pcd.points) == 0:
                    # Initialize combined point cloud with the first frame
                    combined_pcd.points = pcd_frame.points
                    combined_pcd.colors = pcd_frame.colors
                else:
                    # Align the captured frame with the accumulated point cloud
                    pcd_frame_aligned = point_cloud_alignment.align_point_clouds(pcd_frame, combined_pcd)
                    combined_pcd += pcd_frame_aligned

                # **Estimate normals for the combined point cloud**
                self.estimate_normals(combined_pcd)

                # Reconstruct geometry (mesh) from the combined point cloud
                mesh, densities = mesh_reconstruction.reconstruct_mesh(combined_pcd)

                # Highlight regions with gaps or ambiguities (sparse areas)
                mesh_with_highlighted_gaps = self.highlight_sparse_regions(mesh, densities)

                # Update the visualizer with the newly reconstructed mesh
                self.update_visualizer(mesh_with_highlighted_gaps)
            else:
                print("No valid point cloud captured, skipping frame.")

            # Free memory
            gc.collect()

        # Destroy the visualizer when the loop is stopped
        self.destroy_visualizer()

