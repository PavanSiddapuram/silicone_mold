import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import os
from typing import Tuple, List
import time


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
        """Group connected faces into regions."""
        if len(face_indices) == 0:
            return []

        # Build adjacency graph for undercut faces
        face_adjacency = mesh.face_adjacency
        undercut_set = set(face_indices)

        # Filter adjacency to only undercut faces
        undercut_adjacency = []
        for edge in face_adjacency:
            if edge[0] in undercut_set and edge[1] in undercut_set:
                undercut_adjacency.append(edge)

        # Find connected components using DFS
        visited = set()
        regions = []

        def dfs(face_idx, current_region):
            if face_idx in visited:
                return
            visited.add(face_idx)
            current_region.append(face_idx)

            # Find adjacent faces
            for edge in undercut_adjacency:
                if edge[0] == face_idx and edge[1] not in visited:
                    dfs(edge[1], current_region)
                elif edge[1] == face_idx and edge[0] not in visited:
                    dfs(edge[0], current_region)

        for face_idx in face_indices:
            if face_idx not in visited:
                region = []
                dfs(face_idx, region)
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
        # Get vertices of the undercut region
        region_faces = mesh.faces[undercut_region]
        region_vertices = mesh.vertices[np.unique(region_faces)]

        # Create boundary of the undercut region
        boundary_edges = self._extract_boundary_edges(mesh, undercut_region)
        boundary_vertices = mesh.vertices[np.unique(boundary_edges)]

        # Offset boundary vertices along extraction direction
        offset_vertices = boundary_vertices + extraction_direction * self.membrane_thickness

        # Create membrane surface by connecting boundary to offset boundary
        membrane_faces = self._create_membrane_faces(boundary_vertices, offset_vertices)

        # Create membrane mesh
        all_vertices = np.vstack([boundary_vertices, offset_vertices])
        membrane_mesh = trimesh.Trimesh(vertices=all_vertices, faces=membrane_faces)

        # Ensure mesh is watertight
        membrane_mesh.fill_holes()

        return membrane_mesh

    def _extract_boundary_edges(self, mesh: trimesh.Trimesh, face_indices: np.ndarray) -> np.ndarray:
        """Extract boundary edges of a face region."""
        region_faces = mesh.faces[face_indices]
        region_face_set = set(face_indices)

        # Get all edges from the region
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

        return np.array(boundary_edges)

    def _create_membrane_faces(self, bottom_vertices: np.ndarray,
                               top_vertices: np.ndarray) -> np.ndarray:
        """Create faces for the membrane surface."""
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

        return np.array(faces)

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
        # Check for intersections
        if membrane.is_watertight and metamold.is_watertight:
            try:
                intersection = membrane.intersection(metamold)
                if intersection.volume > 0:
                    # Adjust membrane position to avoid intersection
                    membrane = self._resolve_intersection(membrane, metamold)
            except:
                print("Warning: Could not compute intersection, proceeding with original membrane")

        return membrane

    def _resolve_intersection(self, membrane: trimesh.Trimesh,
                              metamold: trimesh.Trimesh) -> trimesh.Trimesh:
        """Resolve intersection between membrane and metamold."""
        # Simple approach: offset membrane vertices slightly
        offset_distance = 0.5  # mm

        # Compute average normal of membrane
        avg_normal = np.mean(membrane.face_normals, axis=0)
        avg_normal = avg_normal / np.linalg.norm(avg_normal)

        # Offset vertices
        offset_vertices = membrane.vertices + avg_normal * offset_distance

        return trimesh.Trimesh(vertices=offset_vertices, faces=membrane.faces)

    def generate_all_membranes(self, draw_direction: np.ndarray) -> Tuple[List[trimesh.Trimesh], List[trimesh.Trimesh]]:
        """
        Generate all secondary membranes for both metamold halves.

        Args:
            draw_direction: Primary draw direction vector

        Returns:
            Tuple of (red_membranes, blue_membranes)
        """
        print("Generating secondary membranes...")

        # Analyze undercuts for red metamold
        red_undercuts = self.analyze_undercuts(self.metamold_red, draw_direction)
        red_extraction_dirs = self.compute_extraction_directions(red_undercuts, self.metamold_red)

        # Analyze undercuts for blue metamold
        blue_undercuts = self.analyze_undercuts(self.metamold_blue, -draw_direction)
        blue_extraction_dirs = self.compute_extraction_directions(blue_undercuts, self.metamold_blue)

        # Generate red membranes
        red_membranes = []
        for i, (region, extraction_dir) in enumerate(zip(red_undercuts, red_extraction_dirs)):
            if len(region) > 0:
                membrane = self.generate_membrane_geometry(region, extraction_dir, self.metamold_red)
                membrane = self.optimize_membrane_placement(membrane, self.metamold_red)

                # Filter out small membranes
                if membrane.area > self.min_membrane_area:
                    red_membranes.append(membrane)

        # Generate blue membranes
        blue_membranes = []
        for i, (region, extraction_dir) in enumerate(zip(blue_undercuts, blue_extraction_dirs)):
            if len(region) > 0:
                membrane = self.generate_membrane_geometry(region, extraction_dir, self.metamold_blue)
                membrane = self.optimize_membrane_placement(membrane, self.metamold_blue)

                # Filter out small membranes
                if membrane.area > self.min_membrane_area:
                    blue_membranes.append(membrane)

        print(f"Generated {len(red_membranes)} red membranes and {len(blue_membranes)} blue membranes")

        return red_membranes, blue_membranes

    def save_membranes(self, red_membranes: List[trimesh.Trimesh],
                       blue_membranes: List[trimesh.Trimesh]):
        """Save all membrane meshes to files."""
        # Save red membranes
        for i, membrane in enumerate(red_membranes):
            membrane_path = os.path.join(self.results_dir, f"red_membrane_{i + 1}.stl")
            membrane.export(membrane_path)

        # Save blue membranes
        for i, membrane in enumerate(blue_membranes):
            membrane_path = os.path.join(self.results_dir, f"blue_membrane_{i + 1}.stl")
            membrane.export(membrane_path)

        print(f"Saved {len(red_membranes)} red membranes and {len(blue_membranes)} blue membranes")

    def visualize_membranes(self, red_membranes: List[trimesh.Trimesh],
                            blue_membranes: List[trimesh.Trimesh]):
        """Visualize metamolds with their secondary membranes."""
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


def generate_secondary_membranes(results_dir: str, draw_direction: np.ndarray):
    """
    Main function to generate secondary membranes for metamold extraction.

    Args:
        results_dir: Directory containing metamold files
        draw_direction: Primary draw direction vector
    """
    # Define file paths
    metamold_red_path = os.path.join(results_dir, "metamold_red.stl")
    metamold_blue_path = os.path.join(results_dir, "metamold_blue.stl")
    original_mesh_path = os.path.join(results_dir, "original_mesh.stl")  # You may need to adjust this

    # Check if files exist
    if not os.path.exists(metamold_red_path):
        print(f"Error: {metamold_red_path} not found")
        return

    if not os.path.exists(metamold_blue_path):
        print(f"Error: {metamold_blue_path} not found")
        return

    # Generate membranes
    generator = SecondaryMembraneGenerator(
        metamold_red_path, metamold_blue_path, original_mesh_path, results_dir
    )

    red_membranes, blue_membranes = generator.generate_all_membranes(draw_direction)

    # Save membranes
    generator.save_membranes(red_membranes, blue_membranes)

    # Visualize results
    generator.visualize_membranes(red_membranes, blue_membranes)

    return red_membranes, blue_membranes


# Example usage (add this to your main.py after the metamold generation):
"""
# Generate secondary membranes
red_membranes, blue_membranes = generate_secondary_membranes(results_dir, draw_direction)

print("Secondary membrane generation complete!")
print(f"Red membranes: {len(red_membranes)}")
print(f"Blue membranes: {len(blue_membranes)}")
"""