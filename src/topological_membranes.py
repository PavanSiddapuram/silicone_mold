import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


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

    # Visualize only the highest difficulty results
    visualize_highest_difficulty_only(mesh, problematic_faces, clusters, mold_problems, angle_threshold)

    return mold_problems


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


def visualize_highest_difficulty_only(mesh, problematic_faces, clusters, mold_problems, angle_threshold):
    """Visualize only the highest difficulty areas from the mold extraction analysis."""

    if not mold_problems:
        print("No problematic areas to visualize.")
        return

    # Filter to only areas with difficulty > 0.9
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

# Example usage with different silicone properties:
# analyze_mold_extractability("model.stl", silicone_stretch_limit=2.5)  # Stiffer silicone
# analyze_mold_extractability("model.stl", silicone_stretch_limit=4.0)  # More flexible silicone