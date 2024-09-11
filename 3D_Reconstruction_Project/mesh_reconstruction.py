import open3d as o3d

class MeshReconstruction:
    def reconstruct_mesh(self, pcd_inlier, depth=6):
        """
        Reconstructs a mesh from the point cloud using Poisson surface reconstruction.

        :param pcd_inlier: The input point cloud.
        :param depth: The depth of reconstruction.
        :return: The reconstructed mesh and its densities.
        """
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_inlier, depth=depth)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_unreferenced_vertices()
        return mesh, densities

