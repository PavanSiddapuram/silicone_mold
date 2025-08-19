import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA


def analyze_mold_extractability_with_membrane(stl_path, reference_normal,
                                              angle_threshold=100, silicone_stretch_limit=3.0,
                                              membrane_thickness=2.0, membrane_resolution=50):
    """
    Analyze STL geometry and generate membranes for problematic regions.

    Parameters:
    - stl_path: Path to STL file
    - reference_normal: Reference direction vector (default: [0, 0, 1])
    - angle_threshold: Minimum angle in degrees to consider faces (default: 100)
    - silicone_stretch_limit: Maximum stretch ratio for silicone (default: 3.0)
    - membrane_thickness: Thickness of the membrane in model units (default: 2.0)
    - membrane_resolution: Resolution for membrane surface generation (default: 50)
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
    clustering = DBSCAN(eps=mesh.scale / 50, min_samples=3)
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

    # Sort by extraction difficulty
    mold_problems.sort(key=lambda x: x['extraction_difficulty'], reverse=True)

    # Print analysis results
    print(f"\n=== MOLD EXTRACTION ANALYSIS WITH MEMBRANE GENERATION ===")
    print(f"Silicone properties assumed:")
    print(f"  - Maximum stretch ratio: {silicone_stretch_limit:.1f}x")
    print(f"  - Shore hardness: A20-A30 (flexible)")
    print(f"  - Tear strength: ~25 N/mm")
    print(f"Membrane properties:")
    print(f"  - Thickness: {membrane_thickness:.1f} units")
    print(f"  - Surface resolution: {membrane_resolution} points")

    print(f"\nFound {len(mold_problems)} problematic regions:")

    for i, problem in enumerate(mold_problems):
        print(f"\n--- Region {i + 1} (Cluster {problem['cluster_id']}) ---")
        print(f"Location: ({problem['center'][0]:.2f}, {problem['center'][1]:.2f}, {problem['center'][2]:.2f})")
        print(f"Bounding box: {problem['bbox_size'][0]:.2f} x {problem['bbox_size'][1]:.2f} x {problem['bbox_size'][2]:.2f}")
        print(f"Max depth: {problem['max_depth']:.2f}")
        print(f"Aspect ratio: {problem['aspect_ratio']:.2f}")
        print(f"Curvature severity: {problem['curvature_severity']:.3f}")
        print(f"Required stretch: {problem['required_stretch']:.2f}x")
        print(f"Extraction difficulty: {problem['extraction_difficulty']:.2f}")

        if problem['extraction_difficulty'] > 0.7:
            print("  ⚠️  HIGH RISK: Membrane generation recommended")
        elif problem['extraction_difficulty'] > 0.4:
            print("  ⚡ MEDIUM RISK: Consider membrane or draft angles")
        else:
            print("  ✅ LOW RISK: Should extract with care")

    # Generate membranes for high-risk regions
    high_risk_regions = [p for p in mold_problems if p['extraction_difficulty'] >= 0.7]

    if high_risk_regions:
        print(f"\n=== GEODESIC PATH & MEMBRANE GENERATION ===")
        print(f"Processing {len(high_risk_regions)} high-risk regions...")

        geodesic_results = []
        membrane_surfaces = []

        for region in high_risk_regions:
            # Find geodesic path
            geodesic_info = find_longest_geodesic_in_region(mesh, region)
            geodesic_results.append(geodesic_info)

            # Generate membrane surface
            if len(geodesic_info['path_coordinates']) > 2:
                membrane = generate_membrane_surface(
                    mesh, geodesic_info, region,
                    membrane_thickness, membrane_resolution, reference_normal
                )
                membrane_surfaces.append(membrane)

                print(f"\nRegion {region['cluster_id']}:")
                print(f"  Geodesic path length: {geodesic_info['path_length']:.2f}")
                print(f"  Path complexity: {geodesic_info['complexity']:.3f}")
                print(f"  Membrane surface area: {membrane['surface_area']:.2f}")
                print(f"  Membrane vertices: {len(membrane['vertices'])}")
                print(f"  Suggested mold split: {membrane['split_recommendation']}")
            else:
                membrane_surfaces.append(None)
                print(f"\nRegion {region['cluster_id']}: Path too short for membrane generation")

        # Visualize with membranes
        visualize_with_membranes(mesh, problematic_faces, clusters, mold_problems,
                                 geodesic_results, membrane_surfaces, angle_threshold)

        return {
            'mold_problems': mold_problems,
            'geodesic_results': geodesic_results,
            'membrane_surfaces': membrane_surfaces,
            'mesh': mesh
        }
    else:
        # Visualize only problematic areas
        visualize_highest_difficulty_only(mesh, problematic_faces, clusters, mold_problems, angle_threshold)
        return {
            'mold_problems': mold_problems,
            'geodesic_results': [],
            'membrane_surfaces': [],
            'mesh': mesh
        }


def generate_membrane_surface(mesh, geodesic_info, region, thickness, resolution, reference_normal):
    """
    Generate a membrane surface that passes through the geodesic path.

    Parameters:
    - mesh: Original trimesh object
    - geodesic_info: Dictionary containing geodesic path information
    - region: Region analysis data
    - thickness: Thickness of the membrane
    - resolution: Number of points for surface generation
    - reference_normal: Reference direction for mold extraction

    Returns:
    - Dictionary containing membrane surface data
    """
    path_coords = geodesic_info['path_coordinates']

    if len(path_coords) < 3:
        return None

    # Step 1: Create a coordinate system along the path
    path_direction = create_path_coordinate_system(path_coords)

    # Step 2: Determine membrane orientation relative to reference normal
    region_face_centers = mesh.triangles_center[region['face_indices']]
    membrane_normal = estimate_membrane_orientation(path_coords, region_face_centers)

    # Step 3: Generate surface points around the geodesic path
    surface_points = generate_surface_around_path(
        path_coords, membrane_normal, thickness, resolution
    )

    # Step 4: Create mesh surface
    membrane_mesh = create_membrane_mesh(surface_points, resolution)

    # Step 5: Calculate properties
    surface_area = calculate_surface_area(membrane_mesh['vertices'], membrane_mesh['faces'])
    split_recommendation = analyze_split_recommendation(geodesic_info, region)

    return {
        'vertices': membrane_mesh['vertices'],
        'faces': membrane_mesh['faces'],
        'surface_area': surface_area,
        'path_coordinates': path_coords,
        'membrane_normal': membrane_normal,
        'reference_normal': reference_normal,
        'thickness': thickness,
        'split_recommendation': split_recommendation,
        'mesh_data': membrane_mesh
    }


def create_path_coordinate_system(path_coords):
    """Create a coordinate system along the geodesic path."""
    n_points = len(path_coords)
    tangent_vectors = np.zeros_like(path_coords)

    # Calculate tangent vectors along the path
    for i in range(n_points):
        if i == 0:
            tangent_vectors[i] = path_coords[i+1] - path_coords[i]
        elif i == n_points - 1:
            tangent_vectors[i] = path_coords[i] - path_coords[i-1]
        else:
            tangent_vectors[i] = path_coords[i+1] - path_coords[i-1]

    # Normalize tangent vectors
    for i in range(n_points):
        norm = np.linalg.norm(tangent_vectors[i])
        if norm > 1e-6:
            tangent_vectors[i] /= norm

    return tangent_vectors


def estimate_membrane_orientation(path_coords, region_centers):
    """Estimate the best orientation for the membrane using PCA."""
    # Combine path and region data
    all_points = np.vstack([path_coords, region_centers])

    # Use PCA to find principal directions
    pca = PCA(n_components=3)
    pca.fit(all_points)

    # The third principal component (smallest variance) is likely the best membrane normal
    membrane_normal = pca.components_[2]

    # Ensure consistent orientation
    path_center = np.mean(path_coords, axis=0)
    region_center = np.mean(region_centers, axis=0)

    # Orient normal toward the problematic region
    toward_region = region_center - path_center
    if np.dot(membrane_normal, toward_region) < 0:
        membrane_normal = -membrane_normal

    return membrane_normal


def generate_surface_around_path(path_coords, membrane_normal, thickness, resolution):
    """Generate surface points around the geodesic path."""
    n_path_points = len(path_coords)

    # Create perpendicular vectors to the membrane normal
    # Use Gram-Schmidt to create an orthogonal basis
    arbitrary_vector = np.array([1, 0, 0])
    if abs(np.dot(membrane_normal, arbitrary_vector)) > 0.9:
        arbitrary_vector = np.array([0, 1, 0])

    perp1 = arbitrary_vector - np.dot(arbitrary_vector, membrane_normal) * membrane_normal
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(membrane_normal, perp1)

    # Generate surface points
    surface_points = []

    # Parameters for surface generation
    width_steps = resolution // 10  # Width resolution
    length_steps = resolution  # Length resolution along path

    # Interpolate path to get more points
    if n_path_points > 2:
        # Create parameter values for interpolation
        t_original = np.linspace(0, 1, n_path_points)
        t_new = np.linspace(0, 1, length_steps)

        # Interpolate each coordinate
        from scipy.interpolate import interp1d
        interp_func = interp1d(t_original, path_coords, axis=0, kind='cubic')
        interpolated_path = interp_func(t_new)
    else:
        interpolated_path = path_coords
        length_steps = len(path_coords)

    # Generate surface around each path point
    for i, path_point in enumerate(interpolated_path):
        # Adaptive width based on position along path (wider in middle, narrower at ends)
        width_factor = 1.0 - abs(2 * i / length_steps - 1)  # Bell curve shape
        current_thickness = thickness * (0.3 + 0.7 * width_factor)

        # Generate points across the width
        for j in range(width_steps):
            if width_steps == 1:
                width_param = 0
            else:
                width_param = (j / (width_steps - 1)) * 2 - 1  # -1 to 1

            # Create surface point
            offset = width_param * current_thickness * 0.5
            surface_point = path_point + offset * perp1
            surface_points.append(surface_point)

    return np.array(surface_points)


def create_membrane_mesh(surface_points, resolution):
    """Create a mesh from surface points."""
    n_points = len(surface_points)

    if n_points < 4:
        # Not enough points for a surface
        return {
            'vertices': surface_points,
            'faces': np.array([]),
            'edges': np.array([])
        }

    # Estimate grid dimensions
    width_steps = max(3, resolution // 10)
    length_steps = n_points // width_steps

    if length_steps < 2:
        # Linear arrangement - create a simple strip
        faces = []
        for i in range(n_points - 1):
            # Create degenerate triangles for visualization
            faces.append([i, i, (i + 1) % n_points])

        return {
            'vertices': surface_points,
            'faces': np.array(faces) if faces else np.array([]),
            'edges': np.array([[i, i+1] for i in range(n_points-1)])
        }

    # Create grid-based triangulation
    faces = []
    vertices_reshaped = surface_points.reshape((length_steps, width_steps, 3))

    for i in range(length_steps - 1):
        for j in range(width_steps - 1):
            # Get indices for current quad
            idx1 = i * width_steps + j
            idx2 = i * width_steps + (j + 1)
            idx3 = (i + 1) * width_steps + j
            idx4 = (i + 1) * width_steps + (j + 1)

            # Create two triangles from quad
            faces.append([idx1, idx2, idx3])
            faces.append([idx2, idx4, idx3])

    return {
        'vertices': surface_points,
        'faces': np.array(faces),
        'edges': np.array([])
    }


def calculate_surface_area(vertices, faces):
    """Calculate the surface area of a triangulated mesh."""
    if len(faces) == 0:
        return 0.0

    total_area = 0.0
    for face in faces:
        if len(face) >= 3:
            # Get vertices of the triangle
            v1, v2, v3 = vertices[face[:3]]

            # Calculate area using cross product
            edge1 = v2 - v1
            edge2 = v3 - v1
            cross_product = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(cross_product)
            total_area += area

    return total_area


def analyze_split_recommendation(geodesic_info, region):
    """Analyze how the membrane should split the mold."""
    path_length = geodesic_info['path_length']
    complexity = geodesic_info['complexity']
    difficulty = region['extraction_difficulty']

    if difficulty > 0.9 and path_length > region['bbox_size'].max():
        return "Two-part split: Cut along entire membrane surface"
    elif difficulty > 0.7:
        return "Partial split: Create relief cuts at key points"
    elif complexity > 0.5:
        return "Multi-point split: Several small relief cuts"
    else:
        return "Minimal intervention: Single relief cut"


def find_longest_geodesic_in_region(mesh, region):
    """Find the longest geodesic path within a problematic region."""
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
            'end_vertex': None,
            'path_coordinates': np.array([])
        }

    # Create subgraph for this region
    vertex_map = {v: i for i, v in enumerate(region_vertices)}
    region_coords = mesh.vertices[region_vertices]

    # Build adjacency matrix for region vertices
    edges = []
    edge_weights = []

    for face_idx in region_face_indices:
        face = mesh.faces[face_idx]
        for i in range(3):
            for j in range(i + 1, 3):
                v1, v2 = face[i], face[j]
                if v1 in vertex_map and v2 in vertex_map:
                    idx1, idx2 = vertex_map[v1], vertex_map[v2]
                    weight = np.linalg.norm(region_coords[idx1] - region_coords[idx2])
                    edges.append((idx1, idx2))
                    edge_weights.append(weight)

    if not edges:
        return {
            'path_vertices': [],
            'path_length': 0.0,
            'complexity': 0.0,
            'start_vertex': None,
            'end_vertex': None,
            'path_coordinates': np.array([])
        }

    # Create sparse adjacency matrix
    n_vertices = len(region_vertices)
    row_indices = [e[0] for e in edges] + [e[1] for e in edges]
    col_indices = [e[1] for e in edges] + [e[0] for e in edges]
    weights = edge_weights + edge_weights

    adjacency_matrix = csr_matrix((weights, (row_indices, col_indices)),
                                  shape=(n_vertices, n_vertices))

    # Find all-pairs shortest paths
    dist_matrix = shortest_path(adjacency_matrix, directed=False)

    # Find the longest shortest path
    max_distance = 0
    best_start, best_end = 0, 0

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if not np.isinf(dist_matrix[i, j]) and dist_matrix[i, j] > max_distance:
                max_distance = dist_matrix[i, j]
                best_start, best_end = i, j

    # Reconstruct path
    G = nx.Graph()
    for edge, weight in zip(edges, edge_weights):
        G.add_edge(edge[0], edge[1], weight=weight)

    try:
        path_indices = nx.shortest_path(G, best_start, best_end, weight='weight')
        path_vertices = [region_vertices[i] for i in path_indices]
        path_coords = mesh.vertices[path_vertices]
        complexity = calculate_path_complexity(path_coords)
    except nx.NetworkXNoPath:
        path_vertices = []
        max_distance = 0.0
        complexity = 0.0
        path_coords = np.array([])

    return {
        'path_vertices': path_vertices,
        'path_length': max_distance,
        'complexity': complexity,
        'start_vertex': region_vertices[best_start] if path_vertices else None,
        'end_vertex': region_vertices[best_end] if path_vertices else None,
        'path_coordinates': path_coords
    }


def calculate_path_complexity(path_coords):
    """Calculate complexity of a path based on curvature."""
    if len(path_coords) < 3:
        return 0.0

    curvatures = []
    for i in range(1, len(path_coords) - 1):
        p1, p2, p3 = path_coords[i - 1], path_coords[i], path_coords[i + 1]
        v1 = p2 - p1
        v2 = p3 - p2

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm > 1e-6 and v2_norm > 1e-6:
            v1_unit = v1 / v1_norm
            v2_unit = v2 / v2_norm
            cross_product = np.cross(v1_unit, v2_unit)
            curvature = np.linalg.norm(cross_product)
            curvatures.append(curvature)

    return np.mean(curvatures) if curvatures else 0.0


def analyze_cluster_geometry(mesh, cluster_faces, cluster_centers, reference_normal, stretch_limit):
    """Analyze geometric properties of a cluster to assess mold extraction difficulty."""
    depths = -np.dot(cluster_centers, reference_normal)
    max_depth = np.max(depths) - np.min(depths)

    bbox_min = np.min(cluster_centers, axis=0)
    bbox_max = np.max(cluster_centers, axis=0)
    bbox_size = bbox_max - bbox_min
    aspect_ratio = np.max(bbox_size) / (np.min(bbox_size) + 1e-6)

    face_normals = mesh.face_normals[cluster_faces]
    normal_variations = []

    for i, center in enumerate(cluster_centers):
        distances = np.linalg.norm(cluster_centers - center, axis=1)
        nearby_mask = distances < np.percentile(distances, 20)

        if np.sum(nearby_mask) > 1:
            nearby_normals = face_normals[nearby_mask]
            mean_normal = np.mean(nearby_normals, axis=0)
            mean_normal = mean_normal / np.linalg.norm(mean_normal)
            dot_products = np.dot(nearby_normals, mean_normal)
            variation = 1 - np.mean(dot_products)
            normal_variations.append(variation)

    curvature_severity = np.mean(normal_variations) if normal_variations else 0
    opening_width = np.min(bbox_size[:2])
    required_stretch = 1 + max_depth / max(opening_width, 0.1)

    depth_factor = min(max_depth / 10.0, 1.0)
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


def visualize_with_membranes(mesh, problematic_faces, clusters, mold_problems,
                             geodesic_results, membrane_surfaces, angle_threshold):
    """Visualize problematic areas with geodesic paths and membrane surfaces."""
    difficulty_threshold = 0.7
    high_difficulty_problems = [p for p in mold_problems if p['extraction_difficulty'] >= difficulty_threshold]

    if not high_difficulty_problems:
        print(f"\nNo areas found with difficulty >= {difficulty_threshold}")
        return

    print(f"\nVisualizing {len(high_difficulty_problems)} regions with membranes")

    # Create PyVista mesh
    pv_faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),
        mesh.faces
    ]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    face_colors = np.zeros(len(mesh.faces))
    for problem in high_difficulty_problems:
        face_colors[problem['face_indices']] = 1.0

    pv_mesh.cell_data["ExtractionDifficulty"] = face_colors

    # Create visualization with 4 subplots
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))

    # Subplot 1: Original problematic areas
    plotter.subplot(0, 0)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1])
    plotter.add_scalar_bar(title="Extraction Difficulty")
    plotter.add_text("Problematic Regions", position='upper_left')

    # Subplot 2: Geodesic paths
    plotter.subplot(0, 1)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1], opacity=0.6)

    colors = ['yellow', 'cyan', 'lime', 'magenta', 'orange', 'white']
    for i, (problem, geodesic) in enumerate(zip(high_difficulty_problems, geodesic_results)):
        if len(geodesic['path_coordinates']) > 1:
            try:
                path_spline = pv.Spline(geodesic['path_coordinates'],
                                        n_points=len(geodesic['path_coordinates']) * 2)
                color = colors[i % len(colors)]
                plotter.add_mesh(path_spline, color=color, line_width=5)
            except:
                # Fallback to polyline
                n_points = len(geodesic['path_coordinates'])
                lines = []
                for j in range(n_points - 1):
                    lines.extend([2, j, j + 1])
                path_polyline = pv.PolyData(geodesic['path_coordinates'], lines=lines)
                color = colors[i % len(colors)]
                plotter.add_mesh(path_polyline, color=color, line_width=5)

    plotter.add_text("Geodesic Paths", position='upper_left')

    # Subplot 3: Membrane surfaces
    plotter.subplot(1, 0)
    plotter.add_mesh(pv_mesh, opacity=0.3, color='lightgray')

    for i, membrane in enumerate(membrane_surfaces):
        if membrane is not None and len(membrane['faces']) > 0:
            # Create membrane mesh for visualization
            membrane_pv_faces = np.hstack([
                np.full((membrane['faces'].shape[0], 1), 3),
                membrane['faces']
            ]).flatten()
            membrane_mesh = pv.PolyData(membrane['vertices'], membrane_pv_faces)

            color = colors[i % len(colors)]
            plotter.add_mesh(membrane_mesh, color=color, opacity=0.7,
                             label=f"Membrane {i+1}")

    plotter.add_text("Generated Membranes", position='upper_left')
    plotter.add_legend()

    # Subplot 4: Combined view with labels
    plotter.subplot(1, 1)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1], opacity=0.5)

    # Add geodesic paths
    for i, (problem, geodesic) in enumerate(zip(high_difficulty_problems, geodesic_results)):
        if len(geodesic['path_coordinates']) > 1:
            try:
                path_spline = pv.Spline(geodesic['path_coordinates'],
                                        n_points=len(geodesic['path_coordinates']) * 2)
                color = colors[i % len(colors)]
                plotter.add_mesh(path_spline, color=color, line_width=3)
            except:
                pass

    # Add membranes
    for i, membrane in enumerate(membrane_surfaces):
        if membrane is not None and len(membrane['faces']) > 0:
            membrane_pv_faces = np.hstack([
                np.full((membrane['faces'].shape[0], 1), 3),
                membrane['faces']
            ]).flatten()
            membrane_mesh = pv.PolyData(membrane['vertices'], membrane_pv_faces)
            color = colors[i % len(colors)]
            plotter.add_mesh(membrane_mesh, color=color, opacity=0.4)

    # Add labels
    for i, problem in enumerate(high_difficulty_problems):
        plotter.add_point_labels([problem['center']], [f"R{i+1}"],
                                 point_size=15, font_size=10)

    plotter.add_text("Complete Analysis\n(Regions + Paths + Membranes)", position='upper_left')

    plotter.show()


def visualize_highest_difficulty_only(mesh, problematic_faces, clusters, mold_problems, angle_threshold):
    """Visualize only the highest difficulty areas from the mold extraction analysis."""
    if not mold_problems:
        print("No problematic areas to visualize.")
        return

    difficulty_threshold = 0.7
    high_difficulty_problems = [p for p in mold_problems if p['extraction_difficulty'] >= difficulty_threshold]

    if not high_difficulty_problems:
        print(f"\nNo areas found with difficulty >= {difficulty_threshold}")
        print("Consider lowering the threshold to see problematic areas.")
        return

    print(f"\nVisualizing {len(high_difficulty_problems)} highest difficulty areas")

    # Create PyVista mesh
    pv_faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),
        mesh.faces
    ]).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, pv_faces)

    face_colors = np.zeros(len(mesh.faces))
    for problem in high_difficulty_problems:
        face_colors[problem['face_indices']] = 1.0

    pv_mesh.cell_data["ExtractionDifficulty"] = face_colors

    # Create visualization
    plotter = pv.Plotter(shape=(1, 2))

    # Left plot: Highest difficulty areas only
    plotter.subplot(0, 0)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1])
    plotter.add_scalar_bar(title="Extraction Difficulty")
    plotter.add_text(f"Highest Difficulty Areas\n(Difficulty >= {difficulty_threshold:.2f})",
                     position='upper_left')

    # Right plot: With labels
    plotter.subplot(0, 1)
    plotter.add_mesh(pv_mesh, scalars="ExtractionDifficulty", show_edges=True,
                     cmap="Reds", clim=[0, 1], opacity=0.8)

    for i, problem in enumerate(high_difficulty_problems):
        risk_level = "HIGH" if problem['extraction_difficulty'] > 0.7 else "MED"
        plotter.add_point_labels([problem['center']], [f"{risk_level}-{i + 1}"],
                                 point_size=20, font_size=12)

    plotter.add_text("Highest Risk Regions", position='upper_left')
    plotter.show()


def export_membrane_stl(membrane_surfaces, output_prefix="membrane"):
    """
    Export generated membrane surfaces as STL files.

    Parameters:
    - membrane_surfaces: List of membrane surface dictionaries
    - output_prefix: Prefix for output filenames

    Returns:
    - List of exported filenames
    """
    exported_files = []

    for i, membrane in enumerate(membrane_surfaces):
        if membrane is not None and len(membrane['faces']) > 0:
            # Create trimesh object
            try:
                membrane_mesh = trimesh.Trimesh(
                    vertices=membrane['vertices'],
                    faces=membrane['faces']
                )

                # Export as STL
                filename = f"{output_prefix}_region_{i+1}.stl"
                membrane_mesh.export(filename)
                exported_files.append(filename)
                print(f"Exported membrane {i+1} to {filename}")

            except Exception as e:
                print(f"Failed to export membrane {i+1}: {e}")

    return exported_files


def generate_mold_split_instructions(mold_problems, membrane_surfaces, geodesic_results):
    """
    Generate detailed instructions for splitting the mold based on membrane analysis.

    Parameters:
    - mold_problems: List of problematic region analyses
    - membrane_surfaces: List of generated membrane surfaces
    - geodesic_results: List of geodesic path results

    Returns:
    - Dictionary with splitting instructions
    """
    instructions = {
        'overview': {},
        'regions': [],
        'manufacturing_notes': [],
        'assembly_notes': []
    }

    high_risk_count = sum(1 for p in mold_problems if p['extraction_difficulty'] > 0.7)
    medium_risk_count = sum(1 for p in mold_problems if 0.4 < p['extraction_difficulty'] <= 0.7)

    instructions['overview'] = {
        'total_problematic_regions': len(mold_problems),
        'high_risk_regions': high_risk_count,
        'medium_risk_regions': medium_risk_count,
        'membranes_generated': len([m for m in membrane_surfaces if m is not None]),
        'recommended_mold_parts': max(2, high_risk_count + 1)
    }

    # Generate instructions for each region
    for i, (problem, membrane, geodesic) in enumerate(zip(mold_problems, membrane_surfaces, geodesic_results)):
        if problem['extraction_difficulty'] > 0.4:  # Only include problematic regions
            region_instructions = {
                'region_id': i + 1,
                'cluster_id': problem['cluster_id'],
                'difficulty': problem['extraction_difficulty'],
                'location': problem['center'].tolist(),
                'dimensions': problem['bbox_size'].tolist(),
                'split_method': 'none'
            }

            if membrane is not None:
                region_instructions.update({
                    'split_method': 'membrane',
                    'membrane_area': membrane['surface_area'],
                    'split_recommendation': membrane['split_recommendation'],
                    'geodesic_length': geodesic['path_length'],
                    'path_complexity': geodesic['complexity']
                })

                # Detailed cutting instructions
                if problem['extraction_difficulty'] > 0.9:
                    cutting_method = "Complete separation: Cut through entire membrane surface"
                elif problem['extraction_difficulty'] > 0.7:
                    cutting_method = "Deep relief cuts: 80% through membrane thickness"
                else:
                    cutting_method = "Shallow relief cuts: 40% through membrane thickness"

                region_instructions['cutting_method'] = cutting_method

            instructions['regions'].append(region_instructions)

    # Manufacturing notes
    instructions['manufacturing_notes'] = [
        "Use CNC machining or EDM for precise membrane cuts",
        "Maintain 0.1mm tolerance for mating surfaces",
        "Add alignment pins for accurate reassembly",
        "Consider draft angles of 1-2° on non-critical surfaces",
        "Use mold release agent on all surfaces",
        f"Recommended silicone: Shore A{20 + min(10, high_risk_count * 2)} hardness"
    ]

    # Assembly notes
    instructions['assembly_notes'] = [
        "Assemble mold parts in numerical order",
        "Apply thin layer of mold release to all interfaces",
        "Use clamps or bolts for consistent pressure",
        "Check alignment before pouring silicone",
        "Allow 24-48 hours curing time before demolding",
        "Remove mold parts in reverse order of assembly"
    ]

    return instructions


def save_analysis_report(analysis_results, output_filename="mold_analysis_report.txt"):
    """
    Save a comprehensive analysis report to a text file.

    Parameters:
    - analysis_results: Dictionary returned by analyze_mold_extractability_with_membrane
    - output_filename: Name of the output report file
    """
    mold_problems = analysis_results['mold_problems']
    geodesic_results = analysis_results['geodesic_results']
    membrane_surfaces = analysis_results['membrane_surfaces']

    # Generate splitting instructions
    split_instructions = generate_mold_split_instructions(
        mold_problems, membrane_surfaces, geodesic_results
    )

    with open(output_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOLD EXTRACTABILITY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        # Overview
        f.write("OVERVIEW\n")
        f.write("-"*40 + "\n")
        overview = split_instructions['overview']
        f.write(f"Total Problematic Regions: {overview['total_problematic_regions']}\n")
        f.write(f"High Risk Regions: {overview['high_risk_regions']}\n")
        f.write(f"Medium Risk Regions: {overview['medium_risk_regions']}\n")
        f.write(f"Membranes Generated: {overview['membranes_generated']}\n")
        f.write(f"Recommended Mold Parts: {overview['recommended_mold_parts']}\n\n")

        # Detailed region analysis
        f.write("DETAILED REGION ANALYSIS\n")
        f.write("-"*40 + "\n")
        for i, problem in enumerate(mold_problems):
            f.write(f"\nRegion {i+1} (Cluster {problem['cluster_id']})\n")
            f.write(f"  Location: ({problem['center'][0]:.2f}, {problem['center'][1]:.2f}, {problem['center'][2]:.2f})\n")
            f.write(f"  Dimensions: {problem['bbox_size'][0]:.2f} x {problem['bbox_size'][1]:.2f} x {problem['bbox_size'][2]:.2f}\n")
            f.write(f"  Max Depth: {problem['max_depth']:.2f}\n")
            f.write(f"  Aspect Ratio: {problem['aspect_ratio']:.2f}\n")
            f.write(f"  Curvature Severity: {problem['curvature_severity']:.3f}\n")
            f.write(f"  Required Stretch: {problem['required_stretch']:.2f}x\n")
            f.write(f"  Extraction Difficulty: {problem['extraction_difficulty']:.3f}\n")

            if problem['extraction_difficulty'] > 0.7:
                f.write("  Status: HIGH RISK - Membrane recommended\n")
            elif problem['extraction_difficulty'] > 0.4:
                f.write("  Status: MEDIUM RISK - Consider relief cuts\n")
            else:
                f.write("  Status: LOW RISK - Extract with care\n")

        # Membrane information
        if membrane_surfaces:
            f.write("\n\nMEMBRANE SURFACES\n")
            f.write("-"*40 + "\n")
            for i, membrane in enumerate(membrane_surfaces):
                if membrane is not None:
                    f.write(f"\nMembrane {i+1}:\n")
                    f.write(f"  Surface Area: {membrane['surface_area']:.2f}\n")
                    f.write(f"  Vertices: {len(membrane['vertices'])}\n")
                    f.write(f"  Faces: {len(membrane['faces'])}\n")
                    f.write(f"  Thickness: {membrane['thickness']:.2f}\n")
                    f.write(f"  Split Recommendation: {membrane['split_recommendation']}\n")

        # Manufacturing instructions
        f.write("\n\nMANUFACTURING INSTRUCTIONS\n")
        f.write("-"*40 + "\n")
        for note in split_instructions['manufacturing_notes']:
            f.write(f"• {note}\n")

        # Assembly instructions
        f.write("\n\nASSEMBLY INSTRUCTIONS\n")
        f.write("-"*40 + "\n")
        for note in split_instructions['assembly_notes']:
            f.write(f"• {note}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"\nAnalysis report saved to {output_filename}")


# # Example usage function
# def example_usage():
#     """
#     Example of how to use the enhanced mold analyzer with membrane generation.
#     """
#     print("Enhanced Mold Analyzer with Membrane Generation")
#     print("=" * 50)
#
#     # Example usage (uncomment and modify path as needed)
#     """
#     # Basic analysis with default settings
#     results = analyze_mold_extractability_with_membrane(
#         stl_path="your_model.stl",
#         angle_threshold=100,
#         silicone_stretch_limit=3.0,
#         membrane_thickness=2.0,
#         membrane_resolution=50
#     )
#
#     # Export membrane STL files
#     if results['membrane_surfaces']:
#         exported_files = export_membrane_stl(results['membrane_surfaces'])
#         print(f"Exported {len(exported_files)} membrane files")
#
#     # Save comprehensive report
#     save_analysis_report(results, "detailed_mold_report.txt")
#
#     # Analysis with different parameters for comparison
#     results_flexible = analyze_mold_extractability_with_membrane(
#         stl_path="your_model.stl",
#         silicone_stretch_limit=4.0,  # More flexible silicone
#         membrane_thickness=1.5,      # Thinner membranes
#         membrane_resolution=75       # Higher resolution
#     )
#     """
#
#     pass
#
#
# if __name__ == "__main__":
#     example_usage()