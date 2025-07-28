import pymeshlab
import os
from pymeshlab import PercentageValue

def repair_and_wrap_mesh(mesh_path: str, output_path: str = None, hole_size: int = 100, merge_threshold: float = 1e-5):
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
    
    print("Starting mesh repair process...")
    
    # Step 1: Basic cleanup
    print("Step 1: Basic cleanup...")
    try:
        ms.apply_filter("meshing_remove_duplicate_faces")
    except:
        print("Remove duplicate faces filter not available")
    
    try:
        ms.apply_filter("meshing_remove_duplicate_vertices")
    except:
        print("Remove duplicate vertices filter not available")
    
    try:
        ms.apply_filter("meshing_remove_unreferenced_vertices")
    except:
        print("Remove unreferenced vertices filter not available")
    
    # Step 2: Merge close vertices
    print("Step 2: Merging close vertices...")
    try:
        ms.apply_filter("meshing_merge_close_vertices", threshold=PercentageValue(merge_threshold))
    except Exception as e:
        print(f"Merge close vertices failed: {e}")
    
    # Step 3: Repair non-manifold elements
    print("Step 3: Repairing non-manifold elements...")
    try:
        ms.apply_filter("meshing_repair_non_manifold_edges")
    except Exception as e:
        print(f"Non-manifold edge repair failed: {e}")
        
    try:
        ms.apply_filter("meshing_repair_non_manifold_vertices")
    except Exception as e:
        print(f"Non-manifold vertex repair failed: {e}")
    
    # Step 4: Re-orient faces - try different possible names
    print("Step 4: Re-orienting faces...")
    face_reoriented = False
    possible_reorient_filters = [
        "meshing_re_orient_faces_coherently",
        "meshing_reorient_faces_coherently", 
        "re_orient_all_faces_coherently",
        "reorient_all_faces_coherently"
    ]
    
    for filter_name in possible_reorient_filters:
        try:
            ms.apply_filter(filter_name)
            print(f"Successfully applied {filter_name}")
            face_reoriented = True
            break
        except:
            continue
    
    if not face_reoriented:
        print("Face reorientation filter not found, skipping...")
    
    # Step 5: Try to close holes
    print("Step 5: Attempting to close holes...")
    holes_closed = False
    possible_hole_filters = [
        "meshing_close_holes",
        "close_holes"
    ]
    
    for filter_name in possible_hole_filters:
        try:
            ms.apply_filter(filter_name, maxholesize=hole_size)
            print(f"Successfully closed holes using {filter_name}")
            holes_closed = True
            break
        except Exception as e:
            print(f"{filter_name} failed: {e}")
            continue
    
    if not holes_closed:
        print("Hole closing failed, trying Poisson reconstruction...")
        try:
            # Try to compute normals first
            normal_computed = False
            possible_normal_filters = [
                "compute_normal_for_point_clouds",
                "compute_normals_for_point_sets",
                "compute_normal_per_vertex"
            ]
            
            for filter_name in possible_normal_filters:
                try:
                    ms.apply_filter(filter_name)
                    normal_computed = True
                    break
                except:
                    continue
            
            if normal_computed:
                # Try Poisson reconstruction
                poisson_filters = [
                    "generate_surface_reconstruction_screened_poisson",
                    "surface_reconstruction_screened_poisson"
                ]
                
                for filter_name in poisson_filters:
                    try:
                        ms.apply_filter(filter_name, depth=8, samplespernode=1.5, pointweight=4.0)
                        print(f"Poisson reconstruction completed using {filter_name}")
                        break
                    except Exception as e:
                        print(f"{filter_name} failed: {e}")
                        continue
            
        except Exception as poisson_error:
            print(f"Poisson reconstruction failed: {poisson_error}")
            print("Proceeding with basic repairs only...")
    
    # Step 6: Final cleanup
    print("Step 6: Final cleanup...")
    try:
        ms.apply_filter("meshing_remove_duplicate_faces")
    except:
        pass
    
    try:
        ms.apply_filter("meshing_remove_unreferenced_vertices")
    except:
        pass
    
    # Default output path
    if output_path is None:
        base, ext = os.path.splitext(mesh_path)
        output_path = base + "_wrapped" + ext
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ms.save_current_mesh(output_path)
    print(f"Repaired mesh saved to: {output_path}")
    return output_path

def list_available_filters():
    """
    Helper function to see what filters are actually available in your PyMeshLab installation.
    """
    ms = pymeshlab.MeshSet()
    print("Available filters:")
    ms.print_filter_list()

def repair_mesh_basic(mesh_path: str, output_path: str = None):
    """
    Most basic mesh repair using only filters that definitely exist.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    
    print("Starting basic mesh repair...")
    
    # Only apply filters that almost certainly exist
    try:
        ms.apply_filter("meshing_remove_duplicate_vertices")
        print("Removed duplicate vertices")
    except Exception as e:
        print(f"Could not remove duplicate vertices: {e}")
    
    try:
        ms.apply_filter("meshing_remove_unreferenced_vertices")
        print("Removed unreferenced vertices")
    except Exception as e:
        print(f"Could not remove unreferenced vertices: {e}")
    
    # Default output path
    if output_path is None:
        base, ext = os.path.splitext(mesh_path)
        output_path = base + "_basic_repair" + ext
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ms.save_current_mesh(output_path)
    print(f"Basic repaired mesh saved to: {output_path}")
    return output_path