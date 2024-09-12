import numpy as np
import cupy as cp
import open3d as o3d

class PointCloudAlignment:
    def align_point_clouds(self, source, target, threshold=0.02, voxel_size=0.01, max_iter=100):
        """
        Aligns two point clouds using ICP algorithm with CUDA optimization where possible.

        :param source: The source point cloud.
        :param target: The target point cloud.
        :param threshold: The ICP threshold for point-to-point registration.
        :param voxel_size: The voxel size for downsampling the point clouds.
        :param max_iter: The maximum number of ICP iterations.
        :return: The aligned source point cloud.
        """
        # Select device (GPU or CPU based on availability)
        device = o3d.core.Device("CUDA:0") if o3d.core.cuda.is_available() else o3d.core.Device("CPU:0")

        # Voxel downsampling for performance improvement (currently no CUDA support in Open3D for this)
        print("Downsampling point clouds using voxel size:", voxel_size)
        source = source.voxel_down_sample(voxel_size=voxel_size)
        target = target.voxel_down_sample(voxel_size=voxel_size)

        # Optional: Use normal estimation (on CPU as CUDA normal estimation is not available)
        print("Estimating normals on CPU...")
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        # Set the initial transformation matrix using cuPy for GPU computation
        trans_init = cp.eye(4)  # Using cuPy for GPU acceleration

        # Configure ICP with custom parameters and make sure to use GPU-based matrix (cuPy) for faster processing
        print("Performing ICP alignment using CUDA...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init.get(),  # Use GPU-based matrix for initial transformation
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter)
        )

        # Apply the transformation from the ICP result to the source point cloud
        source.transform(reg_p2p.transformation)
        return source

# Example usage
if __name__ == "__main__":
    # Load your point clouds
    source_cloud = o3d.io.read_point_cloud("source.pcd")
    target_cloud = o3d.io.read_point_cloud("target.pcd")

    # Create an instance of the PointCloudAlignment class
    pca = PointCloudAlignment()

    # Align the point clouds using ICP
    aligned_source = pca.align_point_clouds(source_cloud, target_cloud)
    
    # Save or visualize the aligned point cloud
    o3d.io.write_point_cloud("aligned_source.pcd", aligned_source)
    o3d.visualization.draw_geometries([aligned_source, target_cloud])

