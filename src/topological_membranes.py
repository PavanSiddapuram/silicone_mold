import trimesh
import numpy as np
import pyvista as pv


def highlight_faces_against_normal(stl_path, reference_normal=np.array([0, 0, 1]), angle_threshold=100):
    """
    Highlight faces that have normals at angles greater than angle_threshold degrees
    relative to the reference normal.

    Parameters:
    - stl_path: Path to STL file
    - reference_normal: Reference direction vector (default: [0, 0, 1])
    - angle_threshold: Minimum angle in degrees to highlight faces (default: 100)
    """
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

    # Convert angle threshold to cosine value
    # cos(100°) ≈ -0.174, cos(90°) = 0
    angle_threshold_rad = np.radians(angle_threshold)
    cos_threshold = np.cos(angle_threshold_rad)

    # Mark faces that have angle > threshold degrees
    # For angles > 100°, dot_product < cos(100°) ≈ -0.174
    faces_to_highlight = dot_products < cos_threshold

    # Create PyVista mesh from vertices and faces
    pv_faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),  # Number of points per face (always 3 for triangles)
        mesh.faces
    ]).flatten()

    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    # Add scalar array for coloring: 1 for faces to highlight, 0 otherwise
    scalar = faces_to_highlight.astype(int)
    pv_mesh.cell_data["HighlightedFaces"] = scalar

    # Print statistics
    total_faces = len(faces_to_highlight)
    highlighted_count = np.sum(faces_to_highlight)
    print(f"Total faces: {total_faces}")
    print(f"Faces with angle > {angle_threshold}°: {highlighted_count}")
    print(f"Percentage highlighted: {100 * highlighted_count / total_faces:.1f}%")

    # Visualize using PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="HighlightedFaces", show_edges=True,
                     cmap="coolwarm", clim=[0, 1])
    plotter.add_scalar_bar(title=f"Faces > {angle_threshold}° from Reference")
    plotter.show()

# Example usage:
# highlight_faces_against_normal("your_file.stl", angle_threshold=100)
# highlight_faces_against_normal("your_file.stl", angle_threshold=110)  # Even more restrictive