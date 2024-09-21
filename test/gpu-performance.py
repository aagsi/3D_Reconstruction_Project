import open3d as o3d
import numpy as np
import time

def generate_large_point_cloud_on_cuda(num_points):
    """Generate a large synthetic point cloud on the CUDA device."""
    print("Generating a large point cloud on CUDA:0...")
    # Generate random points and transfer to CUDA
    points = np.random.rand(num_points, 3).astype(np.float32)
    point_cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points, device=o3d.core.Device("CUDA:0")))
    return point_cloud

def voxel_downsample_on_cuda(point_cloud, voxel_size):
    """Perform voxel downsampling on CUDA."""
    print(f"Performing voxel downsampling on CUDA:0 with voxel size {voxel_size}...")
    start_time = time.time()
    # Ensure voxel downsampling is on CUDA
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    end_time = time.time()
    print(f"Voxel downsampling completed in {end_time - start_time:.2f} seconds.")
    return downsampled_point_cloud

def main():
    # Define the number of points and voxel size
    num_points = 10_000_000  # 10 million points
    voxel_size = 0.05  # Voxel size for downsampling

    # Step 1: Generate a large point cloud on CUDA
    point_cloud = generate_large_point_cloud_on_cuda(num_points)

    # Step 2: Perform voxel downsampling on CUDA
    downsampled_point_cloud = voxel_downsample_on_cuda(point_cloud, voxel_size)

    # Step 3: Transfer the downsampled point cloud back to CPU for further use or saving
    print("Transferring the downsampled point cloud back to CPU...")
    downsampled_point_cloud_cpu = downsampled_point_cloud.to(o3d.core.Device("CPU:0"))

    # Convert to legacy Open3D PointCloud for visualization or saving
    downsampled_pcd_legacy = downsampled_point_cloud_cpu.to_legacy_pointcloud()

    # Step 4: Save the downsampled point cloud (optional)
    output_filename = "downsampled_point_cloud_cuda.ply"
    o3d.io.write_point_cloud(output_filename, downsampled_pcd_legacy)
    print(f"Downsampled point cloud saved to '{output_filename}'.")

if __name__ == "__main__":
    main()

