import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import os
from typing import Tuple, List
import time
from collections import deque


class SecondaryMembraneGenerator:
    """
    Generates secondary membranes for metamold extraction based on the Metamolds paper.
    These membranes enable the extraction of silicone molds by creating demolding paths.
    """

    def __init__(self, metamold_red_path: str, metamold_blue_path: str,
                 original_mesh_path: str, results_dir: str):
        self.metamold_red_path = metamold_red_path
        self.metamold_blue_path = metamold_blue_path
        self.original_mesh_path = original_mesh_path
        self.results_dir = results_dir

        # Load meshes
        self.metamold_red = trimesh.load(metamold_red_path)
        self.metamold_blue = trimesh.load(metamold_blue_path)
        self.original_mesh = trimesh.load(original_mesh_path)

        # Membrane parameters
        self.membrane_thickness = 2.0  # mm
        self.extraction_clearance = 1.0  # mm
        self.min_membrane_area = 10.0  # mm²

    def analyze_undercuts(self, mesh: trimesh.Trimesh, draw_direction: np.ndarray) -> List[np.ndarray]:
        """
        Analyze undercuts in the mesh that require secondary membranes.

        Args:
            mesh: The metamold mesh to analyze
            draw_direction: Primary draw direction vector

        Returns:
            List of undercut regions (face indices)
        """
        face_normals = mesh.face_normals
        face_centers = mesh.triangles.mean(axis=1)

        # Calculate angle between face normals and draw direction
        angles = np.arccos(np.clip(np.dot(face_normals, draw_direction), -1, 1))

        # Identify undercut faces (angle > 90 degrees)
        undercut_threshold = np.pi / 2
        undercut_faces = np.where(angles > undercut_threshold)[0]

        # Group connected undercut faces into regions
        undercut_regions = self._group_connected_faces(mesh, undercut_faces)

        return undercut_regions

    def _group_connected_faces(self, mesh: trimesh.Trimesh, face_indices: np.ndarray) -> List[np.ndarray]:
        """Group connected faces into regions using iterative BFS to avoid recursion limits."""
        if len(face_indices) == 0:
            return []

        # Build adjacency graph for undercut faces
        face_adjacency = mesh.face_adjacency
        undercut_set = set(face_indices)

        # Create adjacency dictionary for faster lookup
        adjacency_dict = {}
        for face_idx in face_indices:
            adjacency_dict[face_idx] = []

        # Filter adjacency to only undercut faces and build adjacency dictionary
        for edge in face_adjacency:
            if edge[0] in undercut_set and edge[1] in undercut_set:
                adjacency_dict[edge[0]].append(edge[1])
                adjacency_dict[edge[1]].append(edge[0])

        # Find connected components using iterative BFS
        visited = set()
        regions = []

        def bfs(start_face):
            """Iterative breadth-first search to find connected component."""
            queue = deque([start_face])
            visited.add(start_face)
            component = [start_face]

            while queue:
                current_face = queue.popleft()

                # Check all adjacent faces
                for neighbor in adjacency_dict.get(current_face, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        component.append(neighbor)

            return component

        # Find all connected components
        for face_idx in face_indices:
            if face_idx not in visited:
                region = bfs(face_idx)
                if len(region) > 0:
                    regions.append(np.array(region))

        return regions

    def compute_extraction_directions(self, undercut_regions: List[np.ndarray],
                                      mesh: trimesh.Trimesh) -> List[np.ndarray]:
        """
        Compute optimal extraction directions for each undercut region.

        Args:
            undercut_regions: List of undercut face groups
            mesh: The metamold mesh

        Returns:
            List of extraction direction vectors
        """
        extraction_directions = []

        for region in undercut_regions:
            # Get face normals for this region
            region_normals = mesh.face_normals[region]
            region_centers = mesh.triangles[region].mean(axis=1)

            # Compute average normal (weighted by face area)
            face_areas = mesh.area_faces[region]
            weighted_normal = np.average(region_normals, weights=face_areas, axis=0)

            # Normalize the weighted normal
            norm = np.linalg.norm(weighted_normal)
            if norm > 0:
                weighted_normal = weighted_normal / norm
            else:
                # Fallback to simple average if all areas are zero
                weighted_normal = np.mean(region_normals, axis=0)
                weighted_normal = weighted_normal / np.linalg.norm(weighted_normal)

            # The extraction direction is opposite to the weighted normal
            extraction_direction = -weighted_normal
            extraction_directions.append(extraction_direction)

        return extraction_directions

    def generate_membrane_geometry(self, undercut_region: np.ndarray,
                                   extraction_direction: np.ndarray,
                                   mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Generate the geometry for a secondary membrane.

        Args:
            undercut_region: Face indices of the undercut region
            extraction_direction: Direction for membrane extraction
            mesh: The metamold mesh

        Returns:
            Membrane mesh
        """
        try:
            # Get vertices of the undercut region
            region_faces = mesh.faces[undercut_region]
            region_vertices = mesh.vertices[np.unique(region_faces)]

            # Create boundary of the undercut region
            boundary_edges = self._extract_boundary_edges(mesh, undercut_region)

            if len(boundary_edges) == 0:
                # Fallback: use convex hull of region vertices
                if len(region_vertices) >= 3:
                    hull = ConvexHull(region_vertices[:, :2])  # Project to 2D for boundary
                    boundary_vertices = region_vertices[hull.vertices]
                else:
                    boundary_vertices = region_vertices
            else:
                boundary_vertices = mesh.vertices[np.unique(boundary_edges)]

            # Offset boundary vertices along extraction direction
            offset_vertices = boundary_vertices + extraction_direction * self.membrane_thickness

            # Create membrane surface by connecting boundary to offset boundary
            membrane_faces = self._create_membrane_faces(boundary_vertices, offset_vertices)

            # Create membrane mesh
            all_vertices = np.vstack([boundary_vertices, offset_vertices])
            membrane_mesh = trimesh.Trimesh(vertices=all_vertices, faces=membrane_faces)

            # Ensure mesh has proper normals
            if len(membrane_mesh.faces) > 0:
                try:
                    membrane_mesh.fix_normals()
                except:
                    pass  # Continue even if fix_normals fails

            return membrane_mesh

        except Exception as e:
            print(f"Warning: Could not generate membrane geometry: {e}")
            # Return a simple cube as fallback
            return trimesh.creation.box(extents=[1, 1, 1])

    def _extract_boundary_edges(self, mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
        """Extract boundary edges of a face region."""
        try:
            region_face_set = set(face_indices)

            # Get all edges from the region faces
            edges = []
            for face_idx in face_indices:
                face = mesh.faces[face_idx]
                face_edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
                edges.extend(face_edges)

            # Find boundary edges (edges that appear only once)
            edge_count = {}
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                edge_count[sorted_edge] = edge_count.get(sorted_edge, 0) + 1

            boundary_edges = []
            for edge, count in edge_count.items():
                if count == 1:
                    boundary_edges.append(edge)

            return np.array(boundary_edges) if boundary_edges else np.array([])

        except Exception as e:
            print(f"Warning: Could not extract boundary edges: {e}")
            return np.array([])

    def _create_membrane_faces(self, bottom_vertices: np.ndarray,
                               top_vertices: np.ndarray) -> np.ndarray:
        """Create faces for the membrane surface."""
        if len(bottom_vertices) < 3:
            return np.array([])

        n_vertices = len(bottom_vertices)
        faces = []

        # Create side faces connecting bottom to top
        for i in range(n_vertices):
            next_i = (i + 1) % n_vertices

            # Two triangles per quad
            # Triangle 1: bottom[i], bottom[next_i], top[i]
            faces.append([i, next_i, i + n_vertices])

            # Triangle 2: bottom[next_i], top[next_i], top[i]
            faces.append([next_i, next_i + n_vertices, i + n_vertices])

        return np.array(faces) if faces else np.array([])

    def optimize_membrane_placement(self, membrane: trimesh.Trimesh,
                                    metamold: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Optimize membrane placement to avoid collisions with metamold.

        Args:
            membrane: Initial membrane mesh
            metamold: Metamold mesh

        Returns:
            Optimized membrane mesh
        """
        try:
            # Check for intersections only if both meshes are watertight
            if membrane.is_watertight and metamold.is_watertight:
                try:
                    intersection = membrane.intersection(metamold)
                    if hasattr(intersection, 'volume') and intersection.volume > 0:
                        # Adjust membrane position to avoid intersection
                        membrane = self._resolve_intersection(membrane, metamold)
                except:
                    # If intersection computation fails, just proceed
                    pass
        except Exception as e:
            print(f"Warning: Could not optimize membrane placement: {e}")

        return membrane

    def _resolve_intersection(self, membrane: trimesh.Trimesh,
                              metamold: trimesh.Trimesh) -> trimesh.Trimesh:
        """Resolve intersection between membrane and metamold."""
        try:
            # Simple approach: offset membrane vertices slightly
            offset_distance = 0.5  # mm

            # Compute average normal of membrane
            if len(membrane.face_normals) > 0:
                avg_normal = np.mean(membrane.face_normals, axis=0)
                norm = np.linalg.norm(avg_normal)
                if norm > 0:
                    avg_normal = avg_normal / norm
                else:
                    avg_normal = np.array([0, 0, 1])  # Default up direction
            else:
                avg_normal = np.array([0, 0, 1])

            # Offset vertices
            offset_vertices = membrane.vertices + avg_normal * offset_distance

            return trimesh.Trimesh(vertices=offset_vertices, faces=membrane.faces)

        except Exception as e:
            print(f"Warning: Could not resolve intersection: {e}")
            return membrane

    def generate_all_membranes(self, draw_direction: np.ndarray) -> Tuple[List[trimesh.Trimesh], List[trimesh.Trimesh]]:
        """
        Generate all secondary membranes for both metamold halves.

        Args:
            draw_direction: Primary draw direction vector

        Returns:
            Tuple of (red_membranes, blue_membranes)
        """
        print("Analyzing undercuts...")

        try:
            # Analyze undercuts for red metamold
            red_undercuts = self.analyze_undercuts(self.metamold_red, draw_direction)
            red_extraction_dirs = self.compute_extraction_directions(red_undercuts, self.metamold_red)

            # Analyze undercuts for blue metamold
            blue_undercuts = self.analyze_undercuts(self.metamold_blue, -draw_direction)
            blue_extraction_dirs = self.compute_extraction_directions(blue_undercuts, self.metamold_blue)

            print(f"Found {len(red_undercuts)} red undercut regions and {len(blue_undercuts)} blue undercut regions")

            # Generate red membranes
            red_membranes = []
            for i, (region, extraction_dir) in enumerate(zip(red_undercuts, red_extraction_dirs)):
                if len(region) > 0:
                    try:
                        membrane = self.generate_membrane_geometry(region, extraction_dir, self.metamold_red)
                        membrane = self.optimize_membrane_placement(membrane, self.metamold_red)

                        # Filter out small membranes
                        if membrane.area > self.min_membrane_area:
                            red_membranes.append(membrane)
                            print(f"Generated red membrane {i + 1} with area {membrane.area:.2f}")
                    except Exception as e:
                        print(f"Warning: Could not generate red membrane {i + 1}: {e}")

            # Generate blue membranes
            blue_membranes = []
            for i, (region, extraction_dir) in enumerate(zip(blue_undercuts, blue_extraction_dirs)):
                if len(region) > 0:
                    try:
                        membrane = self.generate_membrane_geometry(region, extraction_dir, self.metamold_blue)
                        membrane = self.optimize_membrane_placement(membrane, self.metamold_blue)

                        # Filter out small membranes
                        if membrane.area > self.min_membrane_area:
                            blue_membranes.append(membrane)
                            print(f"Generated blue membrane {i + 1} with area {membrane.area:.2f}")
                    except Exception as e:
                        print(f"Warning: Could not generate blue membrane {i + 1}: {e}")

            print(f"Successfully generated {len(red_membranes)} red membranes and {len(blue_membranes)} blue membranes")

        except Exception as e:
            print(f"Error in membrane generation: {e}")
            red_membranes, blue_membranes = [], []

        return red_membranes, blue_membranes

    def merge_membranes_into_metamolds(self, red_membranes: List[trimesh.Trimesh],
                                       blue_membranes: List[trimesh.Trimesh]):
        """Merge membranes into the existing metamold files."""
        try:
            # Merge red membranes into red metamold
            if red_membranes:
                print(f"Merging {len(red_membranes)} membranes into red metamold...")
                red_combined = [self.metamold_red] + red_membranes
                merged_red = trimesh.util.concatenate(red_combined)

                # Save the updated red metamold
                red_metamold_path = os.path.join(self.results_dir, "metamold_red.stl")
                merged_red.export(red_metamold_path)
                print(f"Updated red metamold saved to: {red_metamold_path}")

            # Merge blue membranes into blue metamold
            if blue_membranes:
                print(f"Merging {len(blue_membranes)} membranes into blue metamold...")
                blue_combined = [self.metamold_blue] + blue_membranes
                merged_blue = trimesh.util.concatenate(blue_combined)

                # Save the updated blue metamold
                blue_metamold_path = os.path.join(self.results_dir, "metamold_blue.stl")
                merged_blue.export(blue_metamold_path)
                print(f"Updated blue metamold saved to: {blue_metamold_path}")

            print(f"Successfully merged membranes into metamolds!")

        except Exception as e:
            print(f"Error merging membranes into metamolds: {e}")

    def visualize_membranes(self, red_membranes: List[trimesh.Trimesh],
                            blue_membranes: List[trimesh.Trimesh]):
        """Visualize metamolds with their secondary membranes."""
        try:
            # Create scene
            scene = trimesh.Scene()

            # Add metamolds
            self.metamold_red.visual.face_colors = [255, 100, 100, 150]  # Semi-transparent red
            self.metamold_blue.visual.face_colors = [100, 100, 255, 150]  # Semi-transparent blue

            scene.add_geometry(self.metamold_red, node_name="metamold_red")
            scene.add_geometry(self.metamold_blue, node_name="metamold_blue")

            # Add red membranes
            for i, membrane in enumerate(red_membranes):
                membrane.visual.face_colors = [255, 150, 150, 200]  # Light red
                scene.add_geometry(membrane, node_name=f"red_membrane_{i}")

            # Add blue membranes
            for i, membrane in enumerate(blue_membranes):
                membrane.visual.face_colors = [150, 150, 255, 200]  # Light blue
                scene.add_geometry(membrane, node_name=f"blue_membrane_{i}")

            # Show scene
            scene.show()

        except Exception as e:
            print(f"Warning: Could not visualize membranes: {e}")


def generate_secondary_membranes(results_dir: str, draw_direction: np.ndarray, original_mesh_path: str):
    """
    Main function to generate secondary membranes for metamold extraction.

    Args:
        results_dir: Directory containing metamold files
        draw_direction: Primary draw direction vector
        original_mesh_path: Path to the original input mesh
    """
    # Define file paths
    metamold_red_path = os.path.join(results_dir, "metamold_red.stl")
    metamold_blue_path = os.path.join(results_dir, "metamold_blue.stl")

    # Check if files exist
    if not os.path.exists(metamold_red_path):
        print(f"Error: {metamold_red_path} not found")
        return [], []

    if not os.path.exists(metamold_blue_path):
        print(f"Error: {metamold_blue_path} not found")
        return [], []

    try:
        # Generate membranes
        generator = SecondaryMembraneGenerator(
            metamold_red_path, metamold_blue_path, original_mesh_path, results_dir
        )

        red_membranes, blue_membranes = generator.generate_all_membranes(draw_direction)

        # Merge membranes into metamolds instead of saving separately
        generator.merge_membranes_into_metamolds(red_membranes, blue_membranes)

        # Visualize results (commented out to avoid display issues in some environments)
        # generator.visualize_membranes(red_membranes, blue_membranes)

        return red_membranes, blue_membranes

    except Exception as e:
        print(f"Error in secondary membrane generation: {e}")
        return [], []