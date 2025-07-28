import trimesh
import open3d as o3d
import numpy as np
import open3d as o3d

from src.finalize_draw_direction import FinalizeDrawDirection
from src.convex_hull_operations import compute_convex_hull_from_stl
from src.convex_hull_operations import create_mesh, split_convex_hull, display_hull
# from src.split_mesh import extract_unique_vertices_from_faces, closest_distance, face_centroid
from src.split_mesh import split_mesh_faces, display_split_faces, split_mesh_edges, display_split_edges
from src.offset_surface_operations import offset_stl_sdf, mesh_hull_dist, display_offset_surface, split_offset_surface
from src.merge_isolated_regions import cleanup_isolated_regions
from src.ruledSurface import ruledSurface
from src.generate_metamold import generate_metamold_red
from src.generate_metamold import generate_metamold_blue
from src.generate_metamold import validate_metamold_files

from src.topological_membranes import TopologicalMembranes

import time
import sys
import os

"""
STEPS:
0. Load the mesh from a terminal command
1. Compute convex hull of the input mesh 
2. Finalize the draw direction from the candidate directions
3. Split Convex Hull into two parts based on alignment with the draw direction
4. Split mesh faces based on proximity to the convex hull sides
"""
# ! later, try to overload the display function to display everything, that will be much cleaner.

if len(sys.argv) < 3:
    print("Usage: python main.py <mesh_path> <num_vectors>")
    sys.exit(1)

# Set the mesh path and number of candidate vectors
mesh_path = sys.argv[1]
try:
    num_vectors = int(sys.argv[2])
except ValueError:
    print("Error: <num_vectors> must be an integer.")
    sys.exit(1)

start_time = time.time()

""" COMPUTE CONVEX HULL OF THE MESH """

convex_hull, o3dmesh, pcd, convex_hull_path = compute_convex_hull_from_stl(mesh_path)
tri_convex_hull = trimesh.load(convex_hull_path)

# ! Added
""" GET DISTANCE BETWEEN MESH AND ITS CONVEX HULL """

dist = mesh_hull_dist(mesh_path, convex_hull_path)

# ! Added
""" COMPUTE THE OFFSET SURFACE OF THE MESH USING SDF """

offset_distance = dist
# Compute the offset surface of the input mesh
offset_stl_path = offset_stl_sdf(mesh_path, offset_distance)

# Convert offset mesh into trimesh object
offset_mesh = trimesh.load(offset_stl_path)
print(len(offset_mesh.faces))
print(len(offset_mesh.vertices))

# Display the offset surface along with mesh and convex hull
display_offset_surface(offset_stl_path, mesh_path, convex_hull_path)
# offset_mesh.show()

""" FINALIZE THE DRAW DIRECTION """

fd = FinalizeDrawDirection(mesh_path, num_vectors)

candidate_vectors = fd.createCandidateVectors()

draw_direction = fd.computeVisibleAreas(candidate_vectors)
print("Ideal Draw Direction: ", draw_direction)

""" SPLIT CONVEX HULL """

d1_hull_mesh, d2_hull_mesh, d1_aligned_faces, d2_aligned_faces = split_convex_hull(tri_convex_hull, draw_direction)

""" SPLIT MESH FACES """

tri_mesh = trimesh.load(mesh_path)

red_mesh, blue_mesh = split_mesh_faces(tri_mesh, tri_convex_hull, offset_mesh, d1_aligned_faces, d2_aligned_faces)

display_split_faces(red_mesh, blue_mesh)

""" MERGE ISOLATED REGIONS """

merged_red, merged_blue = cleanup_isolated_regions(red_mesh, blue_mesh)

# Save the merged meshes

input_file_name = os.path.splitext(os.path.basename(mesh_path))[0]

# Create the results directory path
results_dir = os.path.join("results", input_file_name)
os.makedirs(results_dir, exist_ok=True)

merged_red_path = os.path.join(results_dir, "merged_red.stl")
merged_blue_path = os.path.join(results_dir, "merged_blue.stl")

merged_red.save(merged_red_path)
merged_blue.save(merged_blue_path)

end_time = time.time()
print(f"Total time taken is {end_time - start_time:.2f} seconds")

""" DISPLAY THE SPLIT MESHES """

display_split_faces(merged_red, merged_blue)

""" CREATE RULED SURFACE AND THE COMBINED PARTING SURFACE """
combined_parting_surface = ruledSurface(
        merged_blue_path, merged_red_path, mesh_path
    )

""" GENERATE THE METAMOLD HALVES """

combined_mesh_path = os.path.join(results_dir, "combined_parting_surface.stl")

print("Generating metamold halves...")

# Generate red metamold and save to file
metamold_red_path = generate_metamold_red(
    combined_mesh_path, merged_blue_path, draw_direction, combined_parting_surface, results_dir
)

# Generate blue metamold and save to file
metamold_blue_path = generate_metamold_blue(
    combined_mesh_path, merged_red_path, draw_direction, combined_parting_surface, results_dir
)

# Validate that the metamold files were created successfully
if not validate_metamold_files(results_dir):
    print("Error: Metamold generation failed. Cannot proceed with secondary membranes.")
    sys.exit(1)

print("Metamold generation completed successfully!")
print(f"Red metamold saved to: {metamold_red_path}")
print(f"Blue metamold saved to: {metamold_blue_path}")

# ! NEW TOPOLOGICAL MEMBRANES CODE

""" TOPOLOGICAL MEMBRANE PROCESSING FUNCTIONS """

def process_single_metamold_membranes(metamold_path, metamold_name, results_dir):
    """
    Process topological membranes for a single metamold
    """
    print(f"\n--- Processing {metamold_name} metamold ---")
    print(f"Analyzing: {metamold_path}")

    try:
        # Initialize processor for this specific metamold
        topo_processor = TopologicalMembranes(metamold_path)

        print(f"Detected genus for {metamold_name}: {topo_processor.genus}")

        if topo_processor.genus > 0:
            print(f"Generating membranes for {metamold_name} (genus {topo_processor.genus})...")

            # Generate membranes specific to this metamold
            membranes = topo_processor.generate_membranes()

            if membranes:
                # Load the original metamold
                original_metamold = trimesh.load(metamold_path)

                # Combine with membranes
                components = [original_metamold]
                membranes_added = 0

                for i, membrane in enumerate(membranes):
                    if membrane is not None and membrane.is_valid:
                        components.append(membrane)
                        membranes_added += 1
                        print(f"  ✓ Added membrane {i + 1} to {metamold_name}")
                    else:
                        print(f"  ✗ Skipped invalid membrane {i + 1} for {metamold_name}")

                if membranes_added > 0:
                    # Combine and save
                    final_metamold = trimesh.util.concatenate(components)
                    output_path = os.path.join(results_dir, f"final_metamold_{metamold_name}_with_membranes.stl")
                    final_metamold.export(output_path)
                    print(f"✓ Saved {metamold_name} with {membranes_added} membranes: {output_path}")
                    return output_path, membranes_added, topo_processor.genus
                else:
                    print(f"No membranes added to {metamold_name}")
                    return metamold_path, 0, topo_processor.genus
            else:
                print(f"No membranes generated for {metamold_name}")
                return metamold_path, 0, topo_processor.genus
        else:
            print(f"{metamold_name} is genus 0 - no membranes needed")
            return metamold_path, 0, 0

    except Exception as e:
        print(f"Error processing {metamold_name}: {e}")
        return metamold_path, 0, 0


def process_topological_membranes_per_metamold(metamold_red_path, metamold_blue_path, results_dir):
    """
    Process topological membranes for each metamold individually
    """
    print("Processing topological membranes per metamold...")

    # Process red metamold
    final_red_path, red_membranes, red_genus = process_single_metamold_membranes(
        metamold_red_path, "red", results_dir
    )

    # Process blue metamold
    final_blue_path, blue_membranes, blue_genus = process_single_metamold_membranes(
        metamold_blue_path, "blue", results_dir
    )

    return final_red_path, final_blue_path, red_membranes, blue_membranes, red_genus, blue_genus


""" GENERATE TOPOLOGICAL MEMBRANES FOR OBJECTS WITH GENUS > 0 """

# Main topological processing section
print("\nProcessing topological membranes per metamold...")

# Process each metamold individually
final_red_path, final_blue_path, red_membranes_count, blue_membranes_count, red_genus, blue_genus = process_topological_membranes_per_metamold(
    metamold_red_path, metamold_blue_path, results_dir
)

# Summary
print(f"\n=== TOPOLOGICAL PROCESSING SUMMARY ===")
print(f"Red metamold:")
print(f"  Genus: {red_genus}")
print(f"  Membranes added: {red_membranes_count}")
print(f"  Final file: {final_red_path}")

print(f"\nBlue metamold:")
print(f"  Genus: {blue_genus}")
print(f"  Membranes added: {blue_membranes_count}")
print(f"  Final file: {final_blue_path}")

total_membranes = red_membranes_count + blue_membranes_count
if total_membranes > 0:
    print(f"\nTotal membranes integrated: {total_membranes}")

    # Generate detailed topology report
    topo_info_path = os.path.join(results_dir, "topology_report_per_metamold.txt")
    with open(topo_info_path, 'w') as f:
        f.write(f"Per-Metamold Topological Analysis Report\n")
        f.write(f"========================================\n\n")
        f.write(f"RED METAMOLD ANALYSIS:\n")
        f.write(f"  Original file: {metamold_red_path}\n")
        f.write(f"  Detected genus: {red_genus}\n")
        f.write(f"  Membranes integrated: {red_membranes_count}\n")
        f.write(f"  Final file: {os.path.basename(final_red_path)}\n\n")

        f.write(f"BLUE METAMOLD ANALYSIS:\n")
        f.write(f"  Original file: {metamold_blue_path}\n")
        f.write(f"  Detected genus: {blue_genus}\n")
        f.write(f"  Membranes integrated: {blue_membranes_count}\n")
        f.write(f"  Final file: {os.path.basename(final_blue_path)}\n\n")

        f.write(f"SUMMARY:\n")
        f.write(f"  Total membranes: {total_membranes}\n")
        f.write(f"  Processing method: Individual analysis per metamold\n")
        f.write(f"  Each metamold analyzed separately for topology\n")

    print(f"✓ Detailed topology report saved: {topo_info_path}")
else:
    print("\nNo topological membranes were needed for either metamold.")

print(f"\nPer-metamold topological processing completed!")
print(f"Final output files:")
print(f"  Red metamold: {final_red_path}")
print(f"  Blue metamold: {final_blue_path}")

# Update the final metamold paths for any downstream processing
final_metamold_red_path = final_red_path
final_metamold_blue_path = final_blue_path

print(f"\nAll processing completed successfully!")
print(f"Final metamold files ready for 3D printing:")
print(f"  Red metamold: {final_metamold_red_path}")
print(f"  Blue metamold: {final_metamold_blue_path}")