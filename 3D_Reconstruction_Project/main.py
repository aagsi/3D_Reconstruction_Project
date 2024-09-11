# Main file to orchestrate the 3D reconstruction processfrom realsense_pipeline import RealSensePipeline

import sys
import os

# Add the directory to Python's search path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import open3d as o3d
import threading

from realsense_pipeline import RealSensePipeline
from pointcloud_capture import PointCloudCapture
from pointcloud_alignment import PointCloudAlignment
from visualizer import Visualizer
from pointcloud_processing import PointCloudProcessing
from normal_estimation import NormalEstimation
from mesh_reconstruction import MeshReconstruction
from mesh_saving import MeshSaving

def main():
    # Initialize classes
    pipeline_manager = RealSensePipeline()
    point_cloud_capture = PointCloudCapture()
    point_cloud_alignment = PointCloudAlignment()
    visualizer = Visualizer()
    point_cloud_processing = PointCloudProcessing()
    normal_estimation = NormalEstimation()
    mesh_reconstruction = MeshReconstruction()
    mesh_saving = MeshSaving()

    # Start RealSense pipeline
    pipeline_manager.start_pipeline()

    # Start visualization
    visualizer.initialize_visualizer()

    # Create combined point cloud to accumulate captured frames
    combined_pcd = o3d.geometry.PointCloud()

    # Main loop for capturing and aligning point clouds
    stop_event = threading.Event()
    scan_thread = threading.Thread(
        target=visualizer.scanning_loop,
        args=(stop_event, pipeline_manager, point_cloud_capture, point_cloud_alignment, combined_pcd)
    )
    scan_thread.start()

    input("Press Enter to stop scanning...\n")
    stop_event.set()
    scan_thread.join()

    # Stop the pipeline and visualizer
    pipeline_manager.stop_pipeline()
    visualizer.destroy_visualizer()

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

