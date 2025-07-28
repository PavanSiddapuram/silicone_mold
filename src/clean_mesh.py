import pymeshlab
import os
from pymeshlab import PercentageValue


def repair_and_wrap_mesh(mesh_path: str, output_path: str = None, hole_size: int = 100, merge_threshold: int = 1):
    """
    Repairs and wraps a mesh using PyMeshLab to approximate Fusion 360's 'Wrap' function.

    Parameters:
    - mesh_path (str): Path to the input mesh (.stl, .obj, etc.)
    - output_path (str): Path to save the repaired mesh. If None, appends '_wrapped' to input name.
    - hole_size (int): Maximum hole size to close (default: 100)
    - merge_threshold (float): Distance threshold to merge close vertices (default: 1e-5)
    """

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Apply repair filters
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_merge_close_vertices", threshold=PercentageValue(merge_threshold))
    ms.apply_filter("meshing_close_holes", maxholesize=hole_size)
    ms.apply_filter("compute_normal_for_point_clouds")

    # Default output path
    if output_path is None:
        base, ext = os.path.splitext(mesh_path)
        output_path = base + "_wrapped" + ext

    ms.save_current_mesh(output_path)
    print(f"Repaired mesh saved to: {output_path}")

    return output_path
