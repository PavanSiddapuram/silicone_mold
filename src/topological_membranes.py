import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx


def analyze_mold_extractability(stl_path, reference_normal=np.array([0, 0, 1]),
                                angle_threshold=100, silicone_stretch_limit=3.0):
    """
    Analyze STL geometry to identify areas that would be problematic for silicone mold extraction.

    Parameters:
    - stl_path: Path to STL file
    - reference_normal: Reference direction vector (default: [0, 0, 1])
    - angle_threshold: Minimum angle in degrees to consider faces (default: 100)
    - silicone_stretch_limit: Maximum stretch ratio for silicone (default: 3.0)
    """
    # Normalize the reference direction
    reference_normal = reference_normal / np.linalg.norm(reference_normal)

    # Load STL using trimesh
    mesh = trimesh.load_mesh(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input STL is not a Trimesh object.")

    # Get face normals and centers
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center

    # Find problematic faces (angle > threshold)
    dot_products = np.dot(face_normals, reference_normal)
    angle_threshold_rad = np.radians(angle_threshold)
    cos_threshold = np.cos(angle_threshold_rad)
    problematic_faces = dot_products < cos_threshold

    if not np.any(problematic_faces):
        print("No problematic faces found.")
        return

    # Get coordinates of problematic face centers
    problem_centers = face_centers[problematic_faces]
    problem_indices = np.where(problematic_faces)[0]

    # Cluster problematic faces into connected regions
    clustering = DBSCAN(eps=mesh.scale / 50, min_samples=3)  # Adaptive eps based on mesh scale
    clusters = clustering.fit_predict(problem_centers)

    # Analyze each cluster
    mold_problems = []

    for cluster_id in set(clusters):
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_mask = clusters == cluster_id
        cluster_faces = problem_indices[cluster_mask]
        cluster_centers = problem_centers[cluster_mask]

        # Calculate bounding box of cluster
        bbox_min = np.min(cluster_centers, axis=0)
        bbox_max = np.max(cluster_centers, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_volume = np.prod(bbox_size)

        # Analyze geometry characteristics
        cluster_analysis = analyze_cluster_geometry(mesh, cluster_faces, cluster_centers,
                                                    reference_normal, silicone_stretch_limit)

        mold_problems.append({
            'cluster_id': cluster_id,
            'num_faces': len(cluster_faces),
            'face_indices': cluster_faces,
            'bbox_size': bbox_size,
            'bbox_volume': bbox_volume,
            'center': np.mean(cluster_centers, axis=0),
            **cluster_analysis
        })

    # Sort by extraction difficulty (combination of factors)
    mold_problems.sort(key=lambda x: x['extraction_difficulty'], reverse=True)

    # Print analysis results
    print(f"\n=== MOLD EXTRACTION ANALYSIS ===")
    print(f"Silicone properties assumed:")
    print(f"  - Maximum stretch ratio: {silicone_stretch_limit:.1f}x")
    print(f"  - Shore hardness: A20-A30 (flexible)")
    print(f"  - Tear strength: ~25 N/mm")

    print(f"\nFound {len(mold_problems)} problematic regions:")

    for i, problem in enumerate(mold_problems):
        print(f"\n--- Region {i + 1} (Cluster {problem['cluster_id']}) ---")
        print(f"Location: ({problem['center'][0]:.2f}, {problem['center'][1]:.2f}, {problem['center'][2]:.2f})")
        print(
            f"Bounding box: {problem['bbox_size'][0]:.2f} x {problem['bbox_size'][1]:.2f} x {problem['bbox_size'][2]:.2f}")
        print(f"Max depth: {problem['max_depth']:.2f}")
        print(f"Aspect ratio: {problem['aspect_ratio']:.2f}")
        print(f"Curvature severity: {problem['curvature_severity']:.3f}")
        print(f"Required stretch: {problem['required_stretch']:.2f}x")
        print(f"Extraction difficulty: {problem['extraction_difficulty']:.2f}")

        if problem['extraction_difficulty'] > 0.7:
            print("  ⚠️  HIGH RISK: Likely extraction problems")
        elif problem['extraction_difficulty'] > 0.4:
            print("  ⚡ MEDIUM RISK: May need draft angles or multi-part mold")
        else:
            print("  ✅ LOW RISK: Should extract with care")

    # Find geodesic paths for high-risk regions
    high_risk_regions = [p for p in mold_problems if p['extraction_difficulty'] >= 0.7]

    geodesic_results = []

    if high_risk_regions:
        print(f"\n=== GEODESIC PATH ANALYSIS ===")
        print(f"Analyzing {len(high_risk_regions)} high-risk regions...")

        # geodesic_results = []
        for region in high_risk_regions:
            geodesic_info = find_longest_geodesic_in_region(mesh, region)
            geodesic_results.append(geodesic_info)

            print(f"\nRegion {region['cluster_id']}:")
            print(f"  Longest geodesic path length: {geodesic_info['path_length']:.2f}")
            print(f"  Path complexity: {geodesic_info['complexity']:.3f}")
            print(f"  Number of path segments: {len(geodesic_info['path_vertices'])}")

        # Visualize with geodesic paths
        visualize_with_geodesic_paths(mesh, problematic_faces, clusters, mold_problems,
                                      geodesic_results, angle_threshold)
    else:
        # Visualize only the highest difficulty results (original function)
        visualize_highest_difficulty_only(mesh, problematic_faces, clusters, mold_problems, angle_threshold)

    # Create and visualize membranes for high-risk regions
    high_risk_regions = [p for p in mold_problems if p['extraction_difficulty'] >= 0.7]
    if high_risk_regions:
        print("\nGenerating extraction membranes for high-risk regions...")
        visualize_with_membranes(mesh, high_risk_regions, geodesic_results, reference_normal)

    return mold_problems


def find_longest_geodesic_in_region(mesh, region):
    """
    Find the longest geodesic path within a problematic region.

    Parameters:
    - mesh: Trimesh object
    - region: Dictionary containing region analysis data

    Returns:
    - Dictionary with geodesic path information
    """
    # Get vertices that belong to this region's faces
    region_face_indices = region['face_indices']
    region_vertices = set()

    for face_idx in region_face_indices:
        region_vertices.update(mesh.faces[face_idx])

    region_vertices = list(region_vertices)

    if len(region_vertices) < 2:
        return {
            'path_vertices': [],
            'path_length': 0.0,
            'complexity': 0.0,
            'start_vertex': None,
            'end_vertex': None
        }

    # Create subgraph for this region
    vertex_map = {v: i for i, v in enumerate(region_vertices)}
    region_coords = mesh.vertices[region_vertices]

    # Build adjacency matrix for region vertices
    edges = []
    edge_weights = []

    for face_idx in region_face_indices:
        face = mesh.faces[face_idx]
        # Add edges between all vertex pairs in each face
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = face[i], face[j]
                if v1 in vertex_map and v2 in vertex_map:
                    idx1, idx2 = vertex_map[v1], vertex_map[v2]
                    # Calculate edge weight (Euclidean distance)
                    weight = np.linalg.norm(region_coords[idx1] - region_coords[idx2])
                    edges.append((idx1, idx2))
                    edge_weights.append(weight)

    if not edges:
        return {
            'path_vertices': [],
            'path_length': 0.0,
            'complexity': 0.0,
            'start_vertex': None,
            'end_vertex': None
        }

    # Create sparse adjacency matrix
    n_vertices = len(region_vertices)
    row_indices = [e[0] for e in edges] + [e[1] for e in edges]
    col_indices = [e[1] for e in edges] + [e[0] for e in edges]
    weights = edge_weights + edge_weights  # Make symmetric

    adjacency_matrix = csr_matrix((weights, (row_indices, col_indices)),
                                  shape=(n_vertices, n_vertices))

    # Find all-pairs shortest paths
    dist_matrix = shortest_path(adjacency_matrix, directed=False)

    # Find the longest shortest path (diameter)
    max_distance = 0
    best_start, best_end = 0, 0

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if not np.isinf(dist_matrix[i, j]) and dist_matrix[i, j] > max_distance:
                max_distance = dist_matrix[i, j]
                best_start, best_end = i, j

    # Reconstruct the longest path using NetworkX for path reconstruction
    G = nx.Graph()
    for edge, weight in zip(edges, edge_weights):
        G.add_edge(edge[0], edge[1], weight=weight)

    try:
        path_indices = nx.shortest_path(G, best_start, best_end, weight='weight')
        path_vertices = [region_vertices[i] for i in path_indices]
        path_coords = mesh.vertices[path_vertices]

        # Calculate path complexity (curvature along path)
        complexity = calculate_path_complexity(path_coords)

    except nx.NetworkXNoPath:
        # If no path found, return empty result
        path_vertices = []
        max_distance = 0.0
        complexity = 0.0

    return {
        'path_vertices': path_vertices,
        'path_length': max_distance,
        'complexity': complexity,
        'start_vertex': region_vertices[best_start] if path_vertices else None,
        'end_vertex': region_vertices[best_end] if path_vertices else None,
        'path_coordinates': path_coords if path_vertices else np.array([])
    }


def calculate_path_complexity(path_coords):
    """
    Calculate complexity of a path based on curvature.

    Parameters:
    - path_coords: Array of 3D coordinates along the path

    Returns:
    - complexity: Float representing path complexity (0-1)
    """
    if len(path_coords) < 3:
        return 0.0

    # Calculate curvature at each point along the path
    curvatures = []

    for i in range(1, len(path_coords) - 1):
        # Get three consecutive points
        p1, p2, p3 = path_coords[i - 1], path_coords[i], path_coords[i + 1]

        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 1e-6 and v2_norm > 1e-6:
            v1_unit = v1 / v1_norm
            v2_unit = v2 / v2_norm

            # Calculate curvature as the change in direction
            cross_product = np.cross(v1_unit, v2_unit)
            curvature = np.linalg.norm(cross_product)
            curvatures.append(curvature)

    # Return mean curvature as complexity measure
    return np.mean(curvatures) if curvatures else 0.0


def analyze_cluster_geometry(mesh, cluster_faces, cluster_centers, reference_normal, stretch_limit):
    """Analyze geometric properties of a cluster to assess mold extraction difficulty."""

    # Calculate depths of faces relative to reference plane
    depths = -np.dot(cluster_centers, reference_normal)
    max_depth = np.max(depths) - np.min(depths)

    # Calculate bounding box aspect ratio
    bbox_min = np.min(cluster_centers, axis=0)
    bbox_max = np.max(cluster_centers, axis=0)
    bbox_size = bbox_max - bbox_min
    aspect_ratio = np.max(bbox_size) / (np.min(bbox_size) + 1e-6)

    # Estimate curvature using face normal variations
    face_normals = mesh.face_normals[cluster_faces]
    normal_variations = []

    for i, center in enumerate(cluster_centers):
        # Find nearby faces
        distances = np.linalg.norm(cluster_centers - center, axis=1)
        nearby_mask = distances < np.percentile(distances, 20)  # Closest 20%

        if np.sum(nearby_mask) > 1:
            nearby_normals = face_normals[nearby_mask]
            # Calculate standard deviation of normal directions
            mean_normal = np.mean(nearby_normals, axis=0)
            mean_normal = mean_normal / np.linalg.norm(mean_normal)
            dot_products = np.dot(nearby_normals, mean_normal)
            variation = 1 - np.mean(dot_products)  # Higher = more curved
            normal_variations.append(variation)

    curvature_severity = np.mean(normal_variations) if normal_variations else 0

    # Estimate required silicone stretch
    # Based on depth and narrowness of features
    opening_width = np.min(bbox_size[:2])  # Assuming Z is up
    required_stretch = 1 + max_depth / max(opening_width, 0.1)

    # Calculate overall extraction difficulty (0-1 scale)
    # Factors: depth, curvature, required stretch, aspect ratio
    depth_factor = min(max_depth / 10.0, 1.0)  # Normalize to reasonable range
    curvature_factor = min(curvature_severity * 10, 1.0)
    stretch_factor = min(required_stretch / stretch_limit, 1.0)
    aspect_factor = min((aspect_ratio - 1) / 10, 1.0)

    extraction_difficulty = (depth_factor * 0.3 +
                             curvature_factor * 0.3 +
                             stretch_factor * 0.3 +
                             aspect_factor * 0.1)

    return {
        'max_depth': max_depth,
        'aspect_ratio': aspect_ratio,
        'curvature_severity': curvature_severity,
        'required_stretch': required_stretch,
        'extraction_difficulty': extraction_difficulty
    }


def visualize_with_geodesic_paths(mesh, problematic_faces, clusters, mold_problems,
                                  geodesic_results, angle_threshold):
    """
    Visualize highest difficulty areas with their longest geodesic paths.
    """
    # Filter to only areas with difficulty > 0.7
    difficulty_threshold = 0.7
    high_difficulty_problems = [p for p in mold_problems if p['extraction_difficulty'] >= difficulty_threshold]

    if not high_difficulty_problems:
        print(f"\nNo areas found with difficulty >= {difficulty_threshold}")
        print("Consider lowering the threshold to see problematic areas.")
        return

    print(f"\nVisualizing {len(high_difficulty_problems)} highest difficulty areas with geodesic paths")

    # Create PyVista mesh
    pv_faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),
        mesh.faces
    ]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    # Create coloring - set all faces to 0 initially, then only color the highest difficulty areas
    face_colors = np.zeros(len(mesh.faces))

    for problem in high_difficulty_problems:
        face_colors[problem['face_indices']] = 1.0  # Set to maximum color value

    pv_mesh.cell_data["ExtractionDifficulty"] = face_colors

    # Create visualization with 3 subplots
    plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

    # Left plot: Highest difficulty areas only
    plotter.subplot(0, 0)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1])
    plotter.add_scalar_bar(title="Extraction Difficulty")
    plotter.add_text(f"High Difficulty Regions\n(Difficulty >= {difficulty_threshold:.1f})",
                     position='upper_left')

    # Middle plot: Regions with labels
    plotter.subplot(0, 1)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1], opacity=0.8)

    # Add labels for highest difficulty regions
    for i, problem in enumerate(high_difficulty_problems):
        risk_level = "HIGH" if problem['extraction_difficulty'] > 0.7 else "MED"
        plotter.add_point_labels([problem['center']], [f"{risk_level}-{i + 1}"],
                                 point_size=20, font_size=12)

    plotter.add_text("Risk Region Labels", position='upper_left')

    # Right plot: Geodesic paths
    plotter.subplot(0, 2)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1], opacity=0.6)

    # Add geodesic paths
    colors = ['yellow', 'cyan', 'lime', 'magenta', 'orange', 'white']

    for i, (problem, geodesic) in enumerate(zip(high_difficulty_problems, geodesic_results)):
        if len(geodesic['path_coordinates']) > 1:
            # Create spline for the geodesic path
            try:
                path_spline = pv.Spline(geodesic['path_coordinates'], n_points=len(geodesic['path_coordinates']) * 2)
                color = colors[i % len(colors)]
                plotter.add_mesh(path_spline, color=color, line_width=5,
                                 label=f"Region {problem['cluster_id']}")
            except:
                # Fallback: create polyline manually
                n_points = len(geodesic['path_coordinates'])
                lines = []
                for j in range(n_points - 1):
                    lines.extend([2, j, j + 1])  # 2 points per line segment

                path_polyline = pv.PolyData(geodesic['path_coordinates'], lines=lines)
                color = colors[i % len(colors)]
                plotter.add_mesh(path_polyline, color=color, line_width=5,
                                 label=f"Region {problem['cluster_id']}")

            # Add start and end points
            if geodesic['start_vertex'] is not None:
                start_coord = mesh.vertices[geodesic['start_vertex']]
                plotter.add_mesh(pv.Sphere(radius=mesh.scale / 100, center=start_coord),
                                 color='green', label=f"Start {i + 1}")

            if geodesic['end_vertex'] is not None:
                end_coord = mesh.vertices[geodesic['end_vertex']]
                plotter.add_mesh(pv.Sphere(radius=mesh.scale / 100, center=end_coord),
                                 color='red', label=f"End {i + 1}")

    plotter.add_text("Longest Geodesic Paths\nin High-Risk Regions", position='upper_left')
    plotter.add_legend()

    plotter.show()


def visualize_highest_difficulty_only(mesh, problematic_faces, clusters, mold_problems, angle_threshold):
    """Visualize only the highest difficulty areas from the mold extraction analysis."""

    if not mold_problems:
        print("No problematic areas to visualize.")
        return

    # Filter to only areas with difficulty > 0.7
    difficulty_threshold = 0.7
    high_difficulty_problems = [p for p in mold_problems if p['extraction_difficulty'] >= difficulty_threshold]

    if not high_difficulty_problems:
        print(f"\nNo areas found with difficulty >= {difficulty_threshold}")
        print("Consider lowering the threshold to see problematic areas.")
        return

    print(
        f"\nVisualizing {len(high_difficulty_problems)} highest difficulty areas (threshold: {difficulty_threshold:.1f})")

    # Create PyVista mesh
    pv_faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),
        mesh.faces
    ]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    # Create coloring - set all faces to 0 initially, then only color the highest difficulty areas
    face_colors = np.zeros(len(mesh.faces))

    for problem in high_difficulty_problems:
        face_colors[problem['face_indices']] = 1.0  # Set to maximum color value

    pv_mesh.cell_data["ExtractionDifficulty"] = face_colors

    # Create visualization
    plotter = pv.Plotter(shape=(1, 2))

    # Left plot: Highest difficulty areas only
    plotter.subplot(0, 0)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1])
    plotter.add_scalar_bar(title="Extraction Difficulty")
    plotter.add_text(
        f"Highest Difficulty Areas Only\n(Angle > {angle_threshold}°, Difficulty >= {difficulty_threshold:.2f})",
        position='upper_left')

    # Right plot: Highest difficulty areas with labels
    plotter.subplot(0, 1)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1], opacity=0.8)

    # Add labels only for highest difficulty regions
    for i, problem in enumerate(high_difficulty_problems):
        risk_level = "HIGH" if problem['extraction_difficulty'] > 0.7 else "MED"
        plotter.add_point_labels([problem['center']], [f"{risk_level}-{i + 1}"],
                                 point_size=20, font_size=12)

    plotter.add_text("Highest Risk Regions Only", position='upper_left')
    plotter.show()


def create_membrane_from_geodesic(mesh, geodesic_info, reference_normal):
    """
    Create a membrane surface along a geodesic path using ray casting and Delaunay triangulation.
    Casts rays from each geodesic point along the reference normal to find intersections with the mesh,
    then creates a Delaunay surface between the original points and intersection points.

    Args:
        mesh (trimesh.Trimesh): The input mesh
        geodesic_info (dict): Geodesic path information from find_longest_geodesic_in_region
        reference_normal (np.array): Reference normal direction for ray casting

    Returns:
        pv.PolyData: Membrane surface as PyVista mesh
    """
    path_coords = geodesic_info['path_coordinates']
    if len(path_coords) < 2:
        return None

    # Normalize reference normal
    reference_normal = -1 * reference_normal / np.linalg.norm(reference_normal)

    # Cast rays from each geodesic point along the reference normal
    membrane_points = []

    # Add original geodesic points
    membrane_points.extend(path_coords)

    # Cast rays and find intersections
    ray_intersections = []

    for point in path_coords:
        # Cast ray in reference normal direction to find intersection
        ray_origins = np.array([point])
        ray_directions = np.array([reference_normal])

        # Find intersections with the mesh
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions
        )

        if len(locations) > 0:
            # Find the closest intersection point that's not too close to the original point
            distances = np.linalg.norm(locations - point, axis=1)
            valid_intersections = distances > mesh.scale * 0.01  # Filter out self-intersections

            if np.any(valid_intersections):
                valid_locations = locations[valid_intersections]
                valid_distances = distances[valid_intersections]

                # Choose the closest valid intersection
                closest_idx = np.argmin(valid_distances)
                intersection_point = valid_locations[closest_idx]
                ray_intersections.append(intersection_point)
            else:
                # If no valid intersection found, project along reference normal
                projection_distance = mesh.scale * 0.5
                projected_point = point + reference_normal * projection_distance
                ray_intersections.append(projected_point)
        else:
            # If no intersection found, project along reference normal
            projection_distance = mesh.scale * 0.5
            projected_point = point + reference_normal * projection_distance
            ray_intersections.append(projected_point)

    # Add intersection points to membrane points
    membrane_points.extend(ray_intersections)

    # Convert to numpy array
    membrane_points = np.array(membrane_points)

    if len(membrane_points) < 4:  # Need at least 4 points for triangulation
        return None

    # Create simple Delaunay triangulation using PyVista
    try:
        # Create point cloud with all points (geodesic + intersections)
        point_cloud = pv.PolyData(membrane_points)

        # Use 2D Delaunay triangulation (project to best-fit plane)
        # Find the best fitting plane using PCA of all points
        centered_points = membrane_points - np.mean(membrane_points, axis=0)
        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)

        # Project to 2D using first two principal components
        points_2d = centered_points @ Vt[:2].T

        # Create 2D Delaunay triangulation
        points_2d_padded = np.column_stack([points_2d, np.zeros(len(points_2d))])
        point_cloud_2d = pv.PolyData(points_2d_padded)
        delaunay_2d = point_cloud_2d.delaunay_2d()

        # Create 3D surface with original points and 2D triangulation
        membrane_surface = pv.PolyData(membrane_points, delaunay_2d.faces)

        return membrane_surface

    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return None


def visualize_with_membranes(mesh, high_difficulty_problems, geodesic_results, reference_normal):
    """
    Visualize the mesh with membrane surfaces along geodesic paths.
    Updated to pass reference_normal to membrane creation.

    Args:
        mesh (trimesh.Trimesh): The input mesh
        high_difficulty_problems (list): List of high-risk regions
        geodesic_results (list): List of geodesic path information
        reference_normal (np.array): Reference normal direction for membrane draping
    """
    # Create PyVista mesh
    pv_faces = np.hstack([np.full((mesh.faces.shape[0], 1), 3), mesh.faces]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    # Create visualization
    plotter = pv.Plotter()

    # Add original mesh
    plotter.add_mesh(pv_mesh, color='lightgray', opacity=0.7, show_edges=True)

    # Add membranes for each high-risk region
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    for i, (problem, geodesic) in enumerate(zip(high_difficulty_problems, geodesic_results)):
        membrane = create_membrane_from_geodesic(mesh, geodesic, reference_normal)
        if membrane is not None:
            color = colors[i % len(colors)]
            plotter.add_mesh(membrane, color=color, opacity=0.8,
                             label=f"Membrane {i + 1}")

    plotter.add_legend()
    plotter.add_text("Mesh with Ray-Cast Delaunay Membranes\n(Along Reference Normal)", position='upper_left')
    plotter.show()

# # Example usage:
# if __name__ == "__main__":
#     # Example usage with different silicone properties:
#     # analyze_mold_extractability("model.stl", silicone_stretch_limit=2.5)  # Stiffer silicone
#     # analyze_mold_extractability("model.stl", silicone_stretch_limit=4.0)  # More flexible silicone
#
#     # Example with your STL file:
#     # results = analyze_mold_extractability("your_model.stl")
#     pass