import trimesh
import numpy as np
import pyvista as pv

def highlight_faces_against_normal(stl_path, reference_normal=np.array([0, 0, 1])):
    # Normalize the reference direction
    reference_normal = reference_normal / np.linalg.norm(reference_normal)

    # Load STL using trimesh
    mesh = trimesh.load_mesh(stl_path)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input STL is not a Trimesh object.")

    # Get face normals
    face_normals = mesh.face_normals  # shape: (n_faces, 3)

    # Dot product with reference direction
    dot_products = np.dot(face_normals, reference_normal)

    # Mark faces that point in the **opposite** direction (angle > 90° ⇒ dot < 0)
    facing_away = dot_products < 0

    # Create PyVista mesh from vertices and faces
    pv_faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),  # Number of points per face (always 3 for triangles)
        mesh.faces
    ]).flatten()

    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    # Add scalar array for coloring: 1 for away-facing, 0 otherwise
    scalar = facing_away.astype(int)
    pv_mesh.cell_data["FacingAway"] = scalar

    # Visualize using PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="FacingAway", show_edges=True, cmap="coolwarm", clim=[0,1])
    plotter.add_scalar_bar(title="Faces Facing Away")
    plotter.show()

# Example usage:
highlight_faces_against_normal(r"C:\Users\hp\OneDrive\Desktop\metamold_red.stl", reference_normal=np.array([-0.37651, 0.69779, -0.60938]))
