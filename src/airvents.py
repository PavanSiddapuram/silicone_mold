import trimesh
import numpy as np
import pyvista as pv
from src.generate_metamold import (
    step1_get_draw_directions,
    step2_calculate_max_extension_distance,
    step3_create_projection_plane_blue,
    #retract_boundary_by_absolute_distance,
    #project_retracted_points_onto_plane,
    step5_create_ruled_surface,
    #retract_boundary_by_absolute_distance,
    step5_create_ruled_surface,
    trimesh_to_pvpoly,
    step3_create_projection_plane_red,
    #delaunay_bridge_surface,
    step5_create_ruled_surface,
)   





import numpy as np

def filter_overlapping_vents(mesh, vent_indices, gravity_dir, radius, max_vents=10):
    """
    Remove overlapping vent candidates, keeping only the highest maxima.
    Limit the number of vents to max_vents.
    """
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    heights = mesh.vertices[vent_indices] @ gravity_dir
    # Sort indices by height descending
    sorted_idx = np.argsort(-heights)
    selected = []
    selected_points = []
    for idx in sorted_idx:
        pt = mesh.vertices[vent_indices[idx]]
        # Check distance to all already selected vents
        if all(np.linalg.norm(pt - sp) > 2*radius for sp in selected_points):
            selected.append(vent_indices[idx])
            selected_points.append(pt)
        if len(selected) >= max_vents:
            break
    return selected

def find_air_vent_candidates(mesh, gravity_dir, neighbor_radius=2):
    """
    Find local maxima on the mesh surface along the gravity direction.
    Returns indices of candidate air vent positions.
    """
    gravity_dir = gravity_dir / np.linalg.norm(gravity_dir)
    heights = mesh.vertices @ gravity_dir
    maxima = []
    for i, v in enumerate(mesh.vertices):
        # Get neighbors (using mesh.vertex_neighbors if available, else fallback)
        try:
            neighbors = mesh.vertex_neighbors[i]
        except AttributeError:
            # Fallback: use k-nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=neighbor_radius+1).fit(mesh.vertices)
            neighbors = nbrs.kneighbors([v], return_distance=False)[0][1:]
        if all(heights[i] > heights[n] for n in neighbors):
            maxima.append(i)
    return maxima

def plot_air_vents(plotter, mesh, vent_indices, color='yellow', radius=5):
    """
    Plot air vents as spheres at the given indices.
    """
    for idx in vent_indices:
        center = mesh.vertices[idx]
        sphere = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(sphere, color=color, opacity=1.0, label='Air Vent')



def generate_metamold_red(mesh_path, mold_half_path, draw_direction):
    """
    Main function to create and visualize the inner wall surface by:
    - Retracting boundary points toward centroid
    - Projecting retracted points onto a plane
    - Creating ruled surface between the two
    """

    try:
        #red_mesh, secondary_membranes = add_secondary_membranes(trimesh.load(mesh_path))
        red_mesh= trimesh.load(mesh_path)
        #red_mesh=trimesh_to_pvpoly(red_mesh)
        merged_red = trimesh.load(mold_half_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Step 1: Get draw directions
    red_draw_direction, blue_draw_direction = step1_get_draw_directions(draw_direction, merged_red)

    # Step 2: Calculate boundary points and centroid
    max_distance, centroid, boundary_points = step2_calculate_max_extension_distance(
        red_mesh, blue_draw_direction)
    

# ...existing code...

    # Step 3: Create projection planes
    plane_origin, plane_normal = step3_create_projection_plane_red(
        centroid=centroid,
        mesh_faces=red_mesh.faces,
        mesh_vertices=red_mesh.vertices,
        max_distance=max_distance,
        extension_factor=0.01
    )

    # Create a second, closer plane for retracted points

    gravity_dir = red_draw_direction  # or red_draw_direction
    vent_indices = find_air_vent_candidates(merged_red, gravity_dir)
    vent_indices = filter_overlapping_vents(merged_red, vent_indices, gravity_dir, radius=20,max_vents=10)
    # Step 4: Retract boundary points inward
    retracted_points = retract_boundary_by_absolute_distance(
        np.array(boundary_points), red_mesh, retract_ratio=0.05)

    # Step 5: Project retracted points onto the closer plane
    retracted_projected = project_retracted_points_onto_plane(
        retracted_points, plane_origin, plane_normal, scale=1.0)

    # Step 7: Project original boundary points onto the farther plane
    boundary_projected = project_retracted_points_onto_plane(
        np.array(boundary_points), plane_origin, plane_normal, scale=1.0)
    
    # Step 8: Create ruled surface for inner wall
    inner_wall = step5_create_ruled_surface(retracted_points, retracted_projected, extension_factor=0.8)

    # Step 8: Create ruled surface for outer wall
    outer_wall = step5_create_ruled_surface(boundary_points, boundary_projected, extension_factor=1.0)

    # upper_wall = delaunay_bridge_surface(projected_boundary, projected_retracted)
    upper_wall = step5_create_ruled_surface(retracted_projected, boundary_projected, extension_factor=1.0)


    # Visualization
    plotter = pv.Plotter()
    plot_air_vents(plotter, merged_red, vent_indices, color='black', radius=20)
    plotter.add_mesh(trimesh_to_pvpoly(merged_red), color='red', opacity=1, show_edges=True, label='Red Mesh')
    plotter.add_mesh(pv.wrap(red_mesh), color='lightgrey', opacity=1)
    plotter.add_mesh(pv.PolyData(retracted_points), color='green', point_size=10,
                     render_points_as_spheres=True, label='Retracted Points')
    plotter.add_mesh(pv.PolyData(retracted_projected), color='blue', point_size=10,
                     render_points_as_spheres=True, label='Projected Points')
    if inner_wall is not None:
        plotter.add_mesh(inner_wall, color='blue', show_edges=True, opacity=1.0, label='Inner Wall')
    plotter.add_mesh(pv.PolyData(boundary_projected), color='red', point_size=10,
                 render_points_as_spheres=True, label='Projected Boundary')
    if outer_wall is not None:
        plotter.add_mesh(outer_wall, color='red', show_edges=True, opacity=1, label='Outer Wall')
    if upper_wall is not None:
        plotter.add_mesh(upper_wall, color='grey', show_edges=True, opacity=1, label='Upper Wall')
    # Visualize secondary membranes
    # if secondary_membranes:
    #     plotter.add_mesh(pv.wrap(secondary_membranes[0]), color='green', opacity=1.0, show_edges=True, label='Secondary Membranes')
    #     for membrane in secondary_membranes[1:]:
    #         plotter.add_mesh(pv.wrap(membrane), color='green', opacity=1.0, show_edges=True)

    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Inner Wall from Retracted Projection')
    plotter.show()




def generate_metamold_blue(mesh_path, mold_half_path, draw_direction):
    """
    Main function to create and visualize the inner wall surface by:
    - Retracting boundary points toward centroid
    - Projecting retracted points onto a plane
    - Creating ruled surface between the two
    """

    try:
        #red_mesh, secondary_membranes = add_secondary_membranes(trimesh.load(mesh_path))
        red_mesh = trimesh.load(mesh_path)
        pv_red_mesh=trimesh_to_pvpoly(red_mesh)
        merged_red = trimesh.load(mold_half_path)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # Step 1: Get draw directions
    red_draw_direction, blue_draw_direction = step1_get_draw_directions(draw_direction, merged_red)

    # Step 2: Calculate boundary points and centroid
    max_distance, centroid, boundary_points = step2_calculate_max_extension_distance(
        red_mesh, red_draw_direction)

# ...existing code...

    # Step 3: Create projection planes
    plane_origin, plane_normal = step3_create_projection_plane_blue(
        centroid=centroid,
        mesh_faces=red_mesh.faces,
        mesh_vertices=red_mesh.vertices,
        max_distance=max_distance,
        extension_factor=0.01
    )
    # plane_normal= -plane_normal  # Invert normal for blue mesh
    # plane_origin = -plane_origin  # Invert origin for blue mesh

    # Create a second, closer plane for retracted points

    gravity_dir = red_draw_direction  # or red_draw_direction
    vent_indices = find_air_vent_candidates(merged_red, gravity_dir)
    vent_indices = filter_overlapping_vents(merged_red, vent_indices, gravity_dir, radius=20,max_vents=10)

    # Step 4: Retract boundary points inward
    retracted_points = retract_boundary_by_absolute_distance(
        np.array(boundary_points), red_mesh, retract_ratio=0.05)

    # Step 5: Project retracted points onto the closer plane
    retracted_projected = project_retracted_points_onto_plane(
        retracted_points, plane_origin, plane_normal, scale=1.0)

    # Step 7: Project original boundary points onto the farther plane
    boundary_projected = project_retracted_points_onto_plane(
        np.array(boundary_points), plane_origin, plane_normal, scale=1.0)
    
    # Step 8: Create ruled surface for inner wall
    inner_wall = step5_create_ruled_surface(retracted_points, retracted_projected, extension_factor=0.8)

    # Step 8: Create ruled surface for outer wall
    outer_wall = step5_create_ruled_surface(boundary_points, boundary_projected, extension_factor=1.0)

    # upper_wall = delaunay_bridge_surface(projected_boundary, projected_retracted)
    upper_wall = step5_create_ruled_surface(retracted_projected, boundary_projected, extension_factor=1.0)


    # Visualization
    plotter = pv.Plotter()
    plot_air_vents(plotter, merged_red, vent_indices, color='black', radius=20)
    plotter.add_mesh(pv.wrap(pv_red_mesh), color='lightgrey', opacity=1)
    plotter.add_mesh(trimesh_to_pvpoly(merged_red), color='red', opacity=1, show_edges=True, label='Red Mesh')
    plotter.add_mesh(pv.wrap(red_mesh), color='lightgrey', opacity=1)
    plotter.add_mesh(pv.PolyData(retracted_points), color='green', point_size=10,
                     render_points_as_spheres=True, label='Retracted Points')
    plotter.add_mesh(pv.PolyData(retracted_projected), color='blue', point_size=10,
                     render_points_as_spheres=True, label='Projected Points')
    if inner_wall is not None:
        plotter.add_mesh(inner_wall, color='blue', show_edges=True, opacity=1.0, label='Inner Wall')
    plotter.add_mesh(pv.PolyData(boundary_projected), color='red', point_size=10,
                 render_points_as_spheres=True, label='Projected Boundary')
    if outer_wall is not None:
        plotter.add_mesh(outer_wall, color='red', show_edges=True, opacity=1, label='Outer Wall')
    if upper_wall is not None:
        plotter.add_mesh(upper_wall, color='grey', show_edges=True, opacity=1, label='Upper Wall')
    # Visualize secondary membranes
    # if secondary_membranes:
    #     plotter.add_mesh(pv.wrap(secondary_membranes[0]), color='black', opacity=1.0, show_edges=True, label='Secondary Membranes')
    #     for membrane in secondary_membranes[1:]:
    #         plotter.add_mesh(pv.wrap(membrane), color='black', opacity=1.0, show_edges=True)

    plotter.add_legend()
    plotter.show_axes()
    plotter.set_background('white')
    plotter.add_title('Inner Wall from Retracted Projection')
    plotter.show()
