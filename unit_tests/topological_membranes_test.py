import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to install and use manifold3d backend for boolean operations
try:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "manifold3d"])
    print("Successfully installed manifold3d backend")
except:
    print("Warning: Could not install manifold3d. Boolean operations may be limited.")
    pass


class MetamoldMembraneDetector:
    def __init__(self, metamold_blue_path, metamold_red_path, cast_object_path=None):
        """
        Initialize with metamold STL files.

        Args:
            metamold_blue_path: Path to blue metamold STL
            metamold_red_path: Path to red metamold STL
            cast_object_path: Optional path to cast object STL (if separate)
        """
        self.metamold_blue = trimesh.load(metamold_blue_path)
        self.metamold_red = trimesh.load(metamold_red_path)
        self.cast_object = None

        if cast_object_path:
            self.cast_object = trimesh.load(cast_object_path)

        # Ensure meshes are properly oriented and watertight
        self.metamold_blue.fix_normals()
        self.metamold_red.fix_normals()

        self.membranes = []

    def extract_cast_object_from_metamolds(self):
        """
        Extract the cast object geometry from the metamolds if not provided separately.
        Uses multiple approaches to find the cast object geometry.
        """
        if self.cast_object is None:
            print("Attempting to extract cast object from metamolds...")

            # First try to install manifold3d for boolean operations
            try:
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "pip", "install", "manifold3d", "-q"],
                               check=False, capture_output=True)
            except:
                pass

            try:
                # Method 1: Try intersection of both metamolds
                print("  Trying intersection method...")
                intersection = self.metamold_blue.intersection(self.metamold_red)
                if intersection.is_volume and intersection.vertices.shape[0] > 0:
                    self.cast_object = intersection
                    print(f"  Successfully extracted cast object with {len(intersection.vertices)} vertices")
                    return
                else:
                    print("  Intersection method failed - no valid volume")

            except Exception as e:
                print(f"  Intersection method error: {e}")

            try:
                # Method 2: Find overlapping regions using point sampling
                print("  Trying point sampling method...")

                # Get bounding box for sampling
                all_vertices = np.vstack([self.metamold_blue.vertices, self.metamold_red.vertices])
                bbox_min = all_vertices.min(axis=0) - 0.1
                bbox_max = all_vertices.max(axis=0) + 0.1

                # Generate sample points
                n_samples = 50000
                sample_points = np.random.uniform(bbox_min, bbox_max, (n_samples, 3))

                # Find points that are inside both metamolds
                print("    Checking point containment...")
                inside_blue = self.metamold_blue.contains(sample_points)
                inside_red = self.metamold_red.contains(sample_points)
                cast_points = sample_points[inside_blue & inside_red]

                if len(cast_points) > 50:  # Need enough points
                    # Create a simple approximation using the overlapping points
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(cast_points)
                        cast_vertices = cast_points[hull.vertices]
                        cast_faces = hull.simplices

                        cast_mesh = trimesh.Trimesh(vertices=cast_vertices, faces=cast_faces)
                        if cast_mesh.is_volume:
                            self.cast_object = cast_mesh
                            print(f"  Successfully created cast object with {len(cast_vertices)} vertices")
                            return
                    except Exception as e:
                        print(f"    ConvexHull creation failed: {e}")

                    # Fallback: create a simple bounding box of the overlapping points
                    cast_min = cast_points.min(axis=0)
                    cast_max = cast_points.max(axis=0)
                    cast_center = (cast_min + cast_max) / 2
                    cast_extents = cast_max - cast_min

                    if np.all(cast_extents > 0):
                        bbox_mesh = trimesh.creation.box(
                            extents=cast_extents,
                            transform=trimesh.transformations.translation_matrix(cast_center)
                        )
                        self.cast_object = bbox_mesh
                        print(f"  Created bounding box approximation of cast object")
                        return
                else:
                    print(f"  Point sampling found only {len(cast_points)} overlapping points")

            except Exception as e:
                print(f"  Point sampling method error: {e}")

            try:
                # Method 3: Use geometric analysis to find interior regions
                print("  Trying geometric analysis method...")

                # Find vertices from blue metamold that are close to red metamold surface
                blue_tree = cKDTree(self.metamold_blue.vertices)
                red_tree = cKDTree(self.metamold_red.vertices)

                # Find vertices that are very close between the metamolds
                cast_vertices = []
                tolerance = 0.01  # 1cm tolerance

                for vertex in self.metamold_blue.vertices:
                    distance, _ = red_tree.query(vertex)
                    if distance < tolerance:
                        cast_vertices.append(vertex)

                if len(cast_vertices) > 10:
                    cast_vertices = np.array(cast_vertices)

                    # Create a simple mesh from these vertices
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(cast_vertices)
                        cast_mesh = trimesh.Trimesh(
                            vertices=cast_vertices[hull.vertices],
                            faces=hull.simplices
                        )

                        if cast_mesh.is_volume:
                            self.cast_object = cast_mesh
                            print(f"  Created cast object from {len(cast_vertices)} interface vertices")
                            return
                    except:
                        pass

            except Exception as e:
                print(f"  Geometric analysis method error: {e}")

            # If all methods fail, create a simple approximation
            print("  All extraction methods failed. Creating simple approximation...")

            # Use the smaller of the two metamolds as an approximation
            blue_volume = self.metamold_blue.volume if self.metamold_blue.is_volume else 0
            red_volume = self.metamold_red.volume if self.metamold_red.is_volume else 0

            if blue_volume > 0 and red_volume > 0:
                if blue_volume < red_volume:
                    self.cast_object = self.metamold_blue.copy()
                else:
                    self.cast_object = self.metamold_red.copy()
                print("  Using smaller metamold as cast object approximation")
            else:
                # Final fallback: use blue metamold
                self.cast_object = self.metamold_blue.copy()
                print("  Using blue metamold as cast object reference (final fallback)")

    def find_parting_surface(self):
        """
        Identify the parting surface between the two metamolds.
        """
        # Find vertices that are very close between the two metamolds
        tree_blue = cKDTree(self.metamold_blue.vertices)
        tree_red = cKDTree(self.metamold_red.vertices)

        # Find matching vertices (within tolerance)
        tolerance = 1e-6
        parting_vertices_blue = []
        parting_vertices_red = []

        for i, v_blue in enumerate(self.metamold_blue.vertices):
            distances, indices = tree_red.query(v_blue, k=1)
            if distances < tolerance:
                parting_vertices_blue.append(i)
                parting_vertices_red.append(indices)

        return np.array(parting_vertices_blue), np.array(parting_vertices_red)

    def build_mesh_graph(self, mesh):
        """
        Build a graph representation of the mesh for pathfinding.
        """
        edges = mesh.edges_unique
        vertices = mesh.vertices

        # Create adjacency matrix
        n_vertices = len(vertices)
        graph = nx.Graph()

        for edge in edges:
            v1, v2 = edge
            distance = np.linalg.norm(vertices[v1] - vertices[v2])
            graph.add_edge(v1, v2, weight=distance)

        return graph

    def find_exterior_boundary_vertices(self, mesh):
        """
        Identify vertices on the exterior boundary of the metamold.
        Uses multiple methods to ensure boundary detection works.
        """
        boundary_vertices = []

        try:
            # Method 1: Find boundary edges (edges that belong to only one face)
            edge_face_count = {}
            for i, face in enumerate(mesh.faces):
                edges_in_face = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
                for edge in edges_in_face:
                    # Normalize edge (smaller index first)
                    edge = tuple(sorted(edge))
                    edge_face_count[edge] = edge_face_count.get(edge, 0) + 1

            # Find edges that appear in only one face (boundary edges)
            boundary_edges = [edge for edge, count in edge_face_count.items() if count == 1]

            if boundary_edges:
                boundary_vertices = list(set([v for edge in boundary_edges for v in edge]))
                print(f"Method 1: Found {len(boundary_vertices)} boundary vertices using edge analysis")
                return boundary_vertices

        except Exception as e:
            print(f"Method 1 boundary detection failed: {e}")

        try:
            # Method 2: Use trimesh's built-in boundary detection
            if hasattr(mesh, 'outline'):
                outline = mesh.outline()
                if outline is not None and hasattr(outline, 'vertices'):
                    # Find vertices in original mesh that are close to outline vertices
                    mesh_tree = cKDTree(mesh.vertices)
                    for outline_vertex in outline.vertices:
                        distances, indices = mesh_tree.query(outline_vertex, k=1)
                        if distances < 1e-6:
                            boundary_vertices.append(indices)

                    boundary_vertices = list(set(boundary_vertices))
                    if boundary_vertices:
                        print(f"Method 2: Found {len(boundary_vertices)} boundary vertices using outline")
                        return boundary_vertices

        except Exception as e:
            print(f"Method 2 boundary detection failed: {e}")

        try:
            # Method 3: Find vertices on the convex hull (exterior vertices)
            from scipy.spatial import ConvexHull
            hull = ConvexHull(mesh.vertices)
            boundary_vertices = list(set(hull.vertices))
            print(f"Method 3: Found {len(boundary_vertices)} boundary vertices using convex hull")
            return boundary_vertices

        except Exception as e:
            print(f"Method 3 boundary detection failed: {e}")

        try:
            # Method 4: Find vertices with highest distance from centroid (exterior-like)
            centroid = np.mean(mesh.vertices, axis=0)
            distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
            # Take top 10% of vertices furthest from centroid
            n_boundary = max(100, len(mesh.vertices) // 10)
            boundary_indices = np.argpartition(distances, -n_boundary)[-n_boundary:]
            boundary_vertices = boundary_indices.tolist()
            print(f"Method 4: Found {len(boundary_vertices)} boundary vertices using distance from centroid")
            return boundary_vertices

        except Exception as e:
            print(f"Method 4 boundary detection failed: {e}")

        # Fallback: Use a sample of all vertices
        n_samples = min(1000, len(mesh.vertices))
        boundary_vertices = np.random.choice(len(mesh.vertices), n_samples, replace=False).tolist()
        print(f"Fallback: Using {len(boundary_vertices)} random vertices as boundary approximation")
        return boundary_vertices

    def compute_escape_paths(self, mesh, interior_vertices, boundary_vertices):
        """
        Compute shortest paths from interior vertices to boundary.
        """
        graph = self.build_mesh_graph(mesh)
        escape_paths = {}
        destination_vertices = {}

        for v_interior in interior_vertices:
            if v_interior not in graph:
                continue

            # Find shortest paths to all boundary vertices
            try:
                paths = nx.single_source_dijkstra_path_length(
                    graph, v_interior, weight='weight')

                # Find closest boundary vertex
                min_dist = float('inf')
                closest_boundary = None

                for v_boundary in boundary_vertices:
                    if v_boundary in paths and paths[v_boundary] < min_dist:
                        min_dist = paths[v_boundary]
                        closest_boundary = v_boundary

                if closest_boundary is not None:
                    # Get the actual path
                    path = nx.shortest_path(graph, v_interior, closest_boundary,
                                            weight='weight')
                    escape_paths[v_interior] = path
                    destination_vertices[v_interior] = closest_boundary

            except nx.NetworkXNoPath:
                continue

        return escape_paths, destination_vertices

    def get_mold_object_interface_edges(self, metamold):
        """
        Find edges along the interface between mold and cast object.
        """
        if self.cast_object is None:
            self.extract_cast_object_from_metamolds()

        # Check if cast object extraction was successful
        if self.cast_object is None or not hasattr(self.cast_object, 'vertices') or len(self.cast_object.vertices) == 0:
            print("Warning: No valid cast object found. Using alternative interface detection...")
            return self._get_interface_edges_alternative(metamold)

        try:
            # Find vertices that are close to the cast object surface
            cast_tree = cKDTree(self.cast_object.vertices)
            interface_vertices = []
            tolerance = 1e-3  # Increased tolerance for robustness

            for i, vertex in enumerate(metamold.vertices):
                distance, _ = cast_tree.query(vertex)
                if distance < tolerance:
                    interface_vertices.append(i)

            # If no interface vertices found, try with larger tolerance
            if len(interface_vertices) == 0:
                tolerance = 1e-2
                print(f"No interface vertices found, trying larger tolerance: {tolerance}")
                for i, vertex in enumerate(metamold.vertices):
                    distance, _ = cast_tree.query(vertex)
                    if distance < tolerance:
                        interface_vertices.append(i)

            # Find edges connecting interface vertices
            interface_edges = []
            for edge in metamold.edges_unique:
                if edge[0] in interface_vertices and edge[1] in interface_vertices:
                    interface_edges.append(edge)

            return interface_edges, interface_vertices

        except Exception as e:
            print(f"Error in interface detection: {e}")
            return self._get_interface_edges_alternative(metamold)

    def _get_interface_edges_alternative(self, metamold):
        """
        Alternative method to find interface edges when cast object extraction fails.
        Uses geometric analysis of the metamold itself.
        """
        print("Using alternative interface detection method...")

        # Find edges that are likely at interfaces by looking at curvature and boundary conditions
        vertices = metamold.vertices
        faces = metamold.faces
        edges = metamold.edges_unique

        # Calculate vertex normals
        vertex_normals = np.zeros_like(vertices)
        for face in faces:
            # Calculate face normal
            v0, v1, v2 = vertices[face]
            face_normal = np.cross(v1 - v0, v2 - v0)
            face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)

            # Add to vertex normals
            vertex_normals[face] += face_normal

        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1)
        vertex_normals = vertex_normals / (norms[:, np.newaxis] + 1e-8)

        # Find vertices with high curvature (potential interface regions)
        interface_vertices = []

        # Method 1: Find vertices where normal changes significantly
        for i, vertex in enumerate(vertices):
            # Find neighboring vertices
            neighbor_indices = []
            for edge in edges:
                if edge[0] == i:
                    neighbor_indices.append(edge[1])
                elif edge[1] == i:
                    neighbor_indices.append(edge[0])

            if len(neighbor_indices) > 0:
                # Calculate average normal difference
                current_normal = vertex_normals[i]
                neighbor_normals = vertex_normals[neighbor_indices]

                # Calculate angular differences
                dots = np.clip(np.dot(neighbor_normals, current_normal), -1, 1)
                angles = np.arccos(dots)
                avg_angle = np.mean(angles)

                # If average angle is high, likely an interface vertex
                if avg_angle > 0.5:  # ~30 degrees
                    interface_vertices.append(i)

        # Method 2: If still no vertices, use boundary detection
        if len(interface_vertices) == 0:
            print("No high-curvature vertices found, using boundary edges...")
            boundary_edges = metamold.edges[trimesh.grouping.group_rows(
                metamold.edges_sorted, require_count=1)]
            interface_vertices = list(np.unique(boundary_edges.flatten()))

        # Find edges connecting interface vertices
        interface_edges = []
        for edge in edges:
            if edge[0] in interface_vertices and edge[1] in interface_vertices:
                interface_edges.append(edge)

        print(
            f"Alternative method found {len(interface_vertices)} interface vertices and {len(interface_edges)} interface edges")
        return interface_edges, interface_vertices

    def check_minimal_surface_intersection(self, metamold, vi, vj, wi, wj):
        """
        Check if the minimal surface bounded by the escape paths intersects the cast object.
        This is a simplified heuristic - creates a triangulated surface and tests intersection.
        """
        if self.cast_object is None:
            return False

        try:
            # Get vertex coordinates
            v_vi = metamold.vertices[vi]
            v_vj = metamold.vertices[vj]
            v_wi = metamold.vertices[wi]
            v_wj = metamold.vertices[wj]

            # Create a simple quadrilateral surface as approximation of minimal surface
            quad_vertices = np.array([v_vi, v_vj, v_wi, v_wj])
            quad_faces = np.array([[0, 1, 2], [1, 2, 3]])

            # Create mesh for the approximated minimal surface
            quad_mesh = trimesh.Trimesh(vertices=quad_vertices, faces=quad_faces)

            # Check intersection with cast object
            intersection = self.cast_object.intersection(quad_mesh)

            # Return True if there's a meaningful intersection
            return intersection.is_volume and intersection.volume > 1e-12

        except Exception as e:
            print(f"Error in intersection test: {e}")
            return False

    def detect_membranes_for_metamold(self, metamold, metamold_name):
        """
        Detect required membranes for a single metamold piece.
        """
        print(f"\nAnalyzing {metamold_name} metamold...")

        # Get interface edges and vertices
        interface_edges, interface_vertices = self.get_mold_object_interface_edges(metamold)
        boundary_vertices = self.find_exterior_boundary_vertices(metamold)

        print(f"Found {len(interface_edges)} interface edges")
        print(f"Found {len(boundary_vertices)} boundary vertices")

        # Compute escape paths
        escape_paths, destination_vertices = self.compute_escape_paths(
            metamold, interface_vertices, boundary_vertices)

        print(f"Computed escape paths for {len(escape_paths)} vertices")

        # Check each interface edge for membrane requirement
        membranes_needed = []

        for edge in interface_edges:
            vi, vj = edge

            # Skip if escape paths not computed for these vertices
            if vi not in destination_vertices or vj not in destination_vertices:
                continue

            wi = destination_vertices[vi]
            wj = destination_vertices[vj]

            # Check if escape paths go to different boundary regions
            if wi != wj:
                # Test minimal surface intersection
                if self.check_minimal_surface_intersection(metamold, vi, vj, wi, wj):
                    membrane_info = {
                        'metamold': metamold_name,
                        'edge': (vi, vj),
                        'edge_coords': (metamold.vertices[vi], metamold.vertices[vj]),
                        'destinations': (wi, wj),
                        'destination_coords': (metamold.vertices[wi], metamold.vertices[wj])
                    }
                    membranes_needed.append(membrane_info)

        print(f"Detected {len(membranes_needed)} membrane requirements")
        return membranes_needed

    def detect_all_membranes(self):
        """
        Detect membranes needed for both metamold pieces.
        """
        print("Starting membrane detection analysis...")

        # Detect membranes for both metamolds
        blue_membranes = self.detect_membranes_for_metamold(self.metamold_blue, "blue")
        red_membranes = self.detect_membranes_for_metamold(self.metamold_red, "red")

        self.membranes = {
            'blue': blue_membranes,
            'red': red_membranes
        }

        total_membranes = len(blue_membranes) + len(red_membranes)
        print(f"\nTotal membranes detected: {total_membranes}")
        print(f"Blue metamold: {len(blue_membranes)} membranes")
        print(f"Red metamold: {len(red_membranes)} membranes")

        return self.membranes

    def visualize_membranes(self, save_plot=True):
        """
        Visualize the detected membranes.
        """
        fig = plt.figure(figsize=(15, 5))

        # Plot blue metamold membranes
        ax1 = fig.add_subplot(131, projection='3d')
        self._plot_metamold_with_membranes(ax1, self.metamold_blue,
                                           self.membranes['blue'], 'blue', 'Blue Metamold')

        # Plot red metamold membranes
        ax2 = fig.add_subplot(132, projection='3d')
        self._plot_metamold_with_membranes(ax2, self.metamold_red,
                                           self.membranes['red'], 'red', 'Red Metamold')

        # Plot both together
        ax3 = fig.add_subplot(133, projection='3d')
        self._plot_combined_view(ax3)

        plt.tight_layout()

        if save_plot:
            plt.savefig('membrane_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualization saved as 'membrane_analysis.png'")

        plt.show()

    def _plot_metamold_with_membranes(self, ax, metamold, membranes, color, title):
        """Helper function to plot a metamold with its membranes."""
        # Plot metamold (simplified wireframe)
        vertices = metamold.vertices
        edges = metamold.edges_unique

        # Sample edges for visualization (to avoid clutter)
        if len(edges) > 1000:
            sample_indices = np.random.choice(len(edges), 1000, replace=False)
            edges = edges[sample_indices]

        for edge in edges:
            v1, v2 = vertices[edge]
            ax.plot3D(*zip(v1, v2), color=color, alpha=0.1, linewidth=0.5)

        # Plot membrane locations
        for i, membrane in enumerate(membranes):
            edge_coords = membrane['edge_coords']
            dest_coords = membrane['destination_coords']

            # Plot the problematic edge in bright color
            ax.plot3D(*zip(edge_coords[0], edge_coords[1]),
                      color='red', linewidth=3, label=f'Membrane {i + 1}' if i < 5 else "")

            # Plot escape paths
            ax.plot3D(*zip(edge_coords[0], dest_coords[0]),
                      color='orange', linewidth=2, alpha=0.7, linestyle='--')
            ax.plot3D(*zip(edge_coords[1], dest_coords[1]),
                      color='orange', linewidth=2, alpha=0.7, linestyle='--')

        ax.set_title(title)
        if membranes:
            ax.legend()

    def _plot_combined_view(self, ax):
        """Helper function to plot combined view."""
        # Plot both metamolds in different colors
        for edges in [self.metamold_blue.edges_unique[:500], self.metamold_red.edges_unique[:500]]:
            metamold = self.metamold_blue if np.array_equal(edges, self.metamold_blue.edges_unique[
                                                                   :500]) else self.metamold_red
            color = 'blue' if metamold == self.metamold_blue else 'red'

            for edge in edges:
                v1, v2 = metamold.vertices[edge]
                ax.plot3D(*zip(v1, v2), color=color, alpha=0.3, linewidth=0.5)

        # Plot all membranes
        all_membranes = self.membranes['blue'] + self.membranes['red']
        for i, membrane in enumerate(all_membranes):
            metamold = self.metamold_blue if membrane['metamold'] == 'blue' else self.metamold_red
            edge_coords = membrane['edge_coords']

            ax.plot3D(*zip(edge_coords[0], edge_coords[1]),
                      color='yellow', linewidth=2, alpha=0.8)

        ax.set_title('Combined View - All Membranes')

    def export_membranes_as_stl(self, output_dir="membranes"):
        """
        Export membrane surfaces as STL files for manufacturing.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        membrane_count = 0

        for metamold_name, membranes in self.membranes.items():
            metamold = self.metamold_blue if metamold_name == 'blue' else self.metamold_red

            for i, membrane in enumerate(membranes):
                try:
                    # Create a simple membrane surface
                    vi, vj = membrane['edge']
                    wi, wj = membrane['destinations']

                    # Get coordinates
                    coords = np.array([
                        metamold.vertices[vi],
                        metamold.vertices[vj],
                        metamold.vertices[wi],
                        metamold.vertices[wj]
                    ])

                    # Create triangular faces for the membrane
                    faces = np.array([[0, 1, 2], [1, 2, 3]])

                    membrane_mesh = trimesh.Trimesh(vertices=coords, faces=faces)

                    # Export as STL
                    filename = f"{output_dir}/membrane_{metamold_name}_{i + 1}.stl"
                    membrane_mesh.export(filename)
                    membrane_count += 1

                except Exception as e:
                    print(f"Error exporting membrane {metamold_name}_{i + 1}: {e}")

        print(f"Exported {membrane_count} membrane STL files to '{output_dir}' directory")

    def generate_report(self):
        """
        Generate a detailed report of the membrane analysis.
        """
        report = []
        report.append("METAMOLD MEMBRANE ANALYSIS REPORT")
        report.append("=" * 40)
        report.append("")

        total_membranes = len(self.membranes['blue']) + len(self.membranes['red'])
        report.append(f"Total membranes required: {total_membranes}")
        report.append("")

        for metamold_name, membranes in self.membranes.items():
            report.append(f"{metamold_name.upper()} METAMOLD:")
            report.append(f"  Membranes required: {len(membranes)}")

            for i, membrane in enumerate(membranes):
                report.append(f"  Membrane {i + 1}:")
                report.append(f"    Edge vertices: {membrane['edge']}")
                report.append(f"    Edge coordinates: {membrane['edge_coords'][0]} -> {membrane['edge_coords'][1]}")
                report.append(f"    Escape destinations: {membrane['destinations']}")
                report.append("")

        report_text = "\n".join(report)

        # Save report to file
        with open("membrane_analysis_report.txt", "w") as f:
            f.write(report_text)

        print(report_text)
        print("Report saved as 'membrane_analysis_report.txt'")

        return report_text


# Usage example
if __name__ == "__main__":
    # Initialize the detector
    detector = MetamoldMembraneDetector(
        metamold_blue_path=r"C:\Users\hp\OneDrive\Desktop\metamold_blue.stl",
        metamold_red_path=r"C:\Users\hp\OneDrive\Desktop\metamold_red.stl"
        # cast_object_path="cast_object.stl"  # Optional if you have it separate
    )

    # Detect all required membranes
    membranes = detector.detect_all_membranes()

    # Generate visualization
    detector.visualize_membranes()

    # Export membrane surfaces as STL files
    detector.export_membranes_as_stl()

    # Generate detailed report
    detector.generate_report()