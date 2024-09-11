import pyrealsense2.pyrealsense2 as rs
import numpy as np
import open3d as o3d
import gc
import threading


import matplotlib.pyplot as plt

import cv2
# Function to start the RealSense pipeline safely
def start_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()

    # Set conservative stream settings to ensure compatibility with D415
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

    try:
        pipeline.start(config)
        return pipeline
    except RuntimeError as e:
        print(f"Failed to start pipeline: {e}")
        device = pipeline.get_active_profile().get_device()
        device.hardware_reset()  # Reset the camera hardware
        pipeline.stop()
        exit(1)

# Capture point cloud from the camera
def capture_point_cloud_from_camera(pipeline, pc):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    points = pc.calculate(depth_frame)
    pc.map_to(color_frame)

    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    pcd_frame = o3d.geometry.PointCloud()
    pcd_frame.points = o3d.utility.Vector3dVector(vtx)
    pcd_frame.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

    pcd_frame = pcd_frame.voxel_down_sample(voxel_size=0.01)
    
    return pcd_frame

# Align point clouds using ICP
def align_point_clouds(source, target):
    threshold = 0.02  # Adjust for registration
    trans_init = np.eye(4)  # Initial transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source.transform(reg_p2p.transformation)
    return source

# The scanning loop in a separate function
def scanning_loop(pipeline, pc, combined_pcd, vis, stop_event):
    while not stop_event.is_set():  # Continue until the event is set
        pcd_frame = capture_point_cloud_from_camera(pipeline, pc)

        if pcd_frame and len(pcd_frame.points) > 0:
            print(f"Captured point cloud with {len(pcd_frame.points)} points.")
            if len(combined_pcd.points) == 0:
                # This ensures that the first point cloud is correctly set
                combined_pcd.points = pcd_frame.points
                combined_pcd.colors = pcd_frame.colors
                vis.add_geometry(combined_pcd)  # Add the first frame to the visualizer
            else:
                # Align new point cloud to the accumulated combined point cloud
                pcd_frame_aligned = align_point_clouds(pcd_frame, combined_pcd)
                combined_pcd += pcd_frame_aligned

            # Update the visualizer in real-time
            vis.update_geometry(combined_pcd)
            vis.poll_events()
            vis.update_renderer()

        else:
            print("No valid point cloud captured, skipping frame.")
        
        # Free up memory after processing each frame
        gc.collect()

# Step 1: Initialize RealSense pipeline and start streaming
pipeline = start_pipeline()

# Initialize point cloud to accumulate point clouds
combined_pcd = o3d.geometry.PointCloud()
pc = rs.pointcloud()

# Create a visualizer for real-time display
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="3D Scan Progress", width=800, height=600)

try:
    print("Starting real-time 3D scanning... Move the camera around the object.")
    print("Press Enter to stop scanning.")

    stop_event = threading.Event()  # This will be used to stop the loop

    # Start the scanning loop in a separate thread
    scan_thread = threading.Thread(target=scanning_loop, args=(pipeline, pc, combined_pcd, vis, stop_event))
    scan_thread.start()

    # Wait for the user to press Enter to stop the scan
    input()  # Wait for user input
    stop_event.set()  # Signal the scanning loop to stop

    # Wait for the thread to finish
    scan_thread.join()

    # Step 3: Save the combined point cloud
    if len(combined_pcd.points) > 0:
        o3d.io.write_point_cloud("captured_data_on_the_fly.ply", combined_pcd)
        print("Saved point cloud to captured_data_on_the_fly.ply")
    else:
        print("No valid frames captured.")
        raise RuntimeError("No valid frames captured.")

finally:
    vis.destroy_window()  # Close the Open3D visualizer
    pipeline.stop()





# Load the point cloud from the file
pcd = o3d.io.read_point_cloud("captured_data_on_the_fly.ply")

# Step 5: Downsample the point cloud to reduce the number of points further
pcd_down = pcd.voxel_down_sample(voxel_size=0.0025)

# Step 6: Remove noise from the point cloud by eliminating outliers
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.2)
pcd_inlier = pcd_down.select_by_index(ind)

# Optionally remove points that are too close together to avoid duplication
pcd_inlier, ind = pcd_inlier.remove_radius_outlier(nb_points=16, radius=0.01)

# Step 7: Estimate normals with an increased search radius for stability
pcd_inlier.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
pcd_inlier.orient_normals_consistent_tangent_plane(100)

# Step 8: Reconstruct the 3D mesh using Poisson surface reconstruction with depth of 6
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_inlier, depth=6)

# Step 9: Post-process the mesh
mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)  # Smoothing the mesh
mesh = mesh.remove_degenerate_triangles()  # Remove degenerate triangles
mesh = mesh.remove_unreferenced_vertices()  # Clean up unused vertices

# Step 10: Save the reconstructed mesh to a PLY file
o3d.io.write_triangle_mesh("output_mesh_on_the_fly.ply", mesh)

# Optional: Save density-colored mesh to another file
densities_np = np.asarray(densities)
density_colors = plt.get_cmap('plasma')((densities_np - densities_np.min()) / (densities_np.max() - densities_np.min()))
mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors[:, :3])

# Save the mesh with color information
o3d.io.write_triangle_mesh("colored_output_mesh_on_the_fly.ply", mesh)

# Free up memory
del pcd, pcd_down, pcd_inlier, mesh, densities_np, density_colors
import gc
gc.collect()

