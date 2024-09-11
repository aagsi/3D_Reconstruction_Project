import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class MeshSaving:
    def save_mesh(self, mesh, densities, filename="output_mesh_on_the_fly.ply", color_filename="colored_output_mesh_on_the_fly.ply"):
        """
        Saves the reconstructed mesh and optionally a density-colored version.

        :param mesh: The mesh to save.
        :param densities: The densities used for coloring the mesh.
        :param filename: The filename for the output mesh.
        :param color_filename: The filename for the colored output mesh.
        """
        o3d.io.write_triangle_mesh(filename, mesh)

        densities_np = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')((densities_np - densities_np.min()) / (densities_np.max() - densities_np.min()))
        mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors[:, :3])
        o3d.io.write_triangle_mesh(color_filename, mesh)

