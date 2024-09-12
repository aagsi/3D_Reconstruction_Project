import sys
import os
import open3d as o3d
import threading

from realsense_pipeline import RealSensePipeline
from pointcloud_capture import PointCloudCapture
from pointcloud_alignment import PointCloudAlignment
from pointcloud_processing import PointCloudProcessingWithCUDA
from normal_estimation import NormalEstimation
from mesh_reconstruction import MeshReconstruction
from mesh_saving import MeshSaving

def main():
    # Initialize classes
    pipeline_manager = RealSensePipeline()
    point_cloud_capture = PointCloudCapture()
    point_cloud_alignment = PointCloudAlignment()
    point_cloud_processing = PointCloudProcessingWithCUDA()
    normal_estimation = NormalEstimation()
    mesh_reconstruction = MeshReconstruction()
    mesh_saving = MeshSaving()

    # Start RealSense pipeline
    pipeline_manager.start_pipeline()

    # Create combined point cloud to accumulate captured frames
    combined_pcd = o3d.geometry.PointCloud()

    # Main loop for capturing and aligning point clouds
    stop_event = threading.Event()

    # Define a simple scanning loop that does not include visualization
    def simple_scanning_loop(stop_event, pipeline_manager, point_cloud_capture, point_cloud_alignment, combined_pcd, mesh_reconstruction):
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

                # Optionally, you can add more processing here

            else:
                print("No valid point cloud captured, skipping frame.")

    scan_thread = threading.Thread(
        target=simple_scanning_loop,
        args=(stop_event, pipeline_manager, point_cloud_capture, point_cloud_alignment, combined_pcd, mesh_reconstruction)
    )

    scan_thread.start()

    input("Press Enter to stop scanning...\n")
    stop_event.set()
    scan_thread.join()

    # Stop the pipeline
    pipeline_manager.stop_pipeline()

    # ** Save the captured point cloud to a file **
    if len(combined_pcd.points) > 0:
        o3d.io.write_point_cloud("captured_data_on_the_fly.ply", combined_pcd)
        print("Saved point cloud to captured_data_on_the_fly.ply")
    else:
        print("No valid frames captured.")
        return  # Exit if no valid point cloud was captured

    # Process point cloud and estimate normals
    pcd = point_cloud_processing.process_point_cloud("captured_data_on_the_fly.ply")
    pcd_with_normals = normal_estimation.estimate_normals(pcd)

    # Reconstruct mesh
    mesh, densities = mesh_reconstruction.reconstruct_mesh(pcd_with_normals)

    # Save mesh
    mesh_saving.save_mesh(mesh, densities)

if __name__ == "__main__":
    main()

