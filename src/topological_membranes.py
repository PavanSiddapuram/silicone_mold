import trimesh
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import os


class TopologicalMembranes:
    def __init__(self, mesh_path):
        """
        Initialize with the original mesh path
        """
        self.mesh_path = mesh_path
        self.mesh = trimesh.load(mesh_path)
        self.genus = self.compute_genus()
        self.tunnel_loops = []
        self.membranes = []

    def compute_genus(self):
        """
        Compute the genus of the mesh using Euler characteristic
        genus = 1 - (V - E + F) / 2
        where V = vertices, E = edges, F = faces
        """
        V = len(self.mesh.vertices)
        F = len(self.mesh.faces)
        E = len(self.mesh.edges)

        euler_char = V - E + F
        genus = 1 - euler_char // 2

        print(f"Mesh statistics: V={V}, E={E}, F={F}")
        print(f"Euler characteristic: {euler_char}")
        print(f"Genus: {genus}")

        return max(0, genus)  # Ensure non-negative genus

    def build_dual_graph(self):
        """
        Build dual graph where faces are nodes and adjacent faces are connected
        """
        face_adjacency = self.mesh.face_adjacency
        num_faces = len(self.mesh.faces)

        # Create adjacency matrix for faces
        adj_matrix = np.zeros((num_faces, num_faces), dtype=bool)
        for edge in face_adjacency:
            adj_matrix[edge[0], edge[1]] = True
            adj_matrix[edge[1], edge[0]] = True

        return adj_matrix

    def find_tunnel_loops_simple(self):
        """
        Simplified approach to find potential tunnel loops using mesh topology
        """
        if self.genus == 0:
            print("Mesh has genus 0, no tunnels detected.")
            return []

        print(f"Detecting tunnel loops for mesh with genus {self.genus}...")

        # Find boundary loops that might represent tunnels
        tunnel_candidates = []

        # Method 1: Find loops in the mesh structure
        edges = self.mesh.edges
        vertices = self.mesh.vertices

        # Build vertex adjacency graph
        vertex_graph = {}
        for edge in edges:
            v1, v2 = edge
            if v1 not in vertex_graph:
                vertex_graph[v1] = set()
            if v2 not in vertex_graph:
                vertex_graph[v2] = set()
            vertex_graph[v1].add(v2)
            vertex_graph[v2].add(v1)

        # Find cycles that could represent tunnel boundaries
        visited = set()
        cycles = []

        def dfs_cycle(start, current, path, target_length=20):
            if len(path) > target_length:
                return
            if len(path) > 3 and current == start:
                cycles.append(path.copy())
                return
            if current in path[:-1]:  # Avoid infinite loops
                return

            if current in vertex_graph:
                for neighbor in vertex_graph[current]:
                    if neighbor not in path or (len(path) > 3 and neighbor == start):
                        path.append(neighbor)
                        dfs_cycle(start, neighbor, path, target_length)
                        path.pop()

        # Sample some vertices to find cycles
        vertex_sample = np.random.choice(len(vertices), min(50, len(vertices)), replace=False)

        for start_vertex in vertex_sample:
            if start_vertex not in visited:
                dfs_cycle(start_vertex, start_vertex, [start_vertex])
                visited.add(start_vertex)

        # Filter and select meaningful cycles
        meaningful_cycles = []
        for cycle in cycles:
            if len(cycle) > 5:  # Minimum cycle length
                cycle_vertices = vertices[cycle]
                # Check if cycle forms a roughly circular path
                center = np.mean(cycle_vertices, axis=0)
                distances = np.linalg.norm(cycle_vertices - center, axis=1)
                if np.std(distances) / np.mean(distances) < 0.5:  # Relatively uniform distances
                    meaningful_cycles.append(cycle)

        # Select the best cycles based on geometric properties
        self.tunnel_loops = meaningful_cycles[:self.genus]  # Limit to genus count

        print(f"Found {len(self.tunnel_loops)} tunnel loop candidates")
        return self.tunnel_loops

    def create_membrane_surface(self, loop_vertices, membrane_thickness=0.001):
        """
        Create a membrane surface for a given loop of vertices
        """
        if len(loop_vertices) < 3:
            return None

        loop_points = self.mesh.vertices[loop_vertices]

        # Find the best fitting plane for the loop
        centroid = np.mean(loop_points, axis=0)

        # Use PCA to find the normal direction
        centered_points = loop_points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[-1]  # Last row is the normal to the best fit plane

        # Create membrane points by projecting loop onto the plane
        membrane_vertices = []
        membrane_faces = []

        # Add center point
        center_idx = len(membrane_vertices)
        membrane_vertices.append(centroid)

        # Add loop points projected to plane
        for point in loop_points:
            projected = point - np.dot(point - centroid, normal) * normal
            membrane_vertices.append(projected)

        # Create triangular faces from center to loop edges
        num_loop_points = len(loop_points)
        for i in range(num_loop_points):
            next_i = (i + 1) % num_loop_points
            # Triangle: center, current point, next point
            membrane_faces.append([center_idx, i + 1, next_i + 1])

        # Create the solid membrane by duplicating and flipping faces
        membrane_vertices = np.array(membrane_vertices)
        membrane_faces = np.array(membrane_faces)

        # Duplicate vertices with slight offset
        offset_vertices = membrane_vertices + normal * membrane_thickness
        all_vertices = np.vstack([membrane_vertices, offset_vertices])

        # Original faces
        all_faces = membrane_faces.tolist()

        # Flipped faces (with vertex offset)
        flipped_faces = membrane_faces + len(membrane_vertices)
        flipped_faces = flipped_faces[:, ::-1]  # Flip orientation
        all_faces.extend(flipped_faces.tolist())

        # Side faces to close the membrane
        num_vertices = len(membrane_vertices)
        for i in range(1, num_vertices):  # Skip center vertex
            next_i = (i % (num_vertices - 1)) + 1
            if next_i == 1:
                next_i = num_vertices - 1

            # Create quads as two triangles
            v1, v2 = i, next_i
            v3, v4 = v1 + num_vertices, v2 + num_vertices

            all_faces.append([v1, v2, v4])
            all_faces.append([v1, v4, v3])

        membrane_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        membrane_mesh.fix_normals()

        return membrane_mesh

    def generate_membranes(self):
        """
        Generate all necessary membranes for the mesh
        """
        if self.genus == 0:
            print("No membranes needed for genus 0 mesh")
            return []

        tunnel_loops = self.find_tunnel_loops_simple()
        membranes = []

        for i, loop in enumerate(tunnel_loops):
            membrane = self.create_membrane_surface(loop)
            if membrane is not None:
                membranes.append(membrane)
                print(f"Created membrane {i + 1} with {len(membrane.vertices)} vertices")

        self.membranes = membranes
        return membranes

    def apply_membranes_to_metamolds(self, metamold_red_path, metamold_blue_path, results_dir):
        """
        Apply membranes to the metamold halves after segmentation
        """
        if not self.membranes:
            print("No membranes to apply")
            return metamold_red_path, metamold_blue_path

        # Load metamolds
        red_metamold = trimesh.load(metamold_red_path)
        blue_metamold = trimesh.load(metamold_blue_path)

        print("Applying membranes to metamolds...")

        # For each membrane, determine which metamold pieces it should affect
        for i, membrane in enumerate(self.membranes):
            membrane_path = os.path.join(results_dir, f"membrane_{i}.stl")
            membrane.export(membrane_path)

            # Simple approach: subtract membrane from both metamolds
            # In practice, you might want more sophisticated logic here
            try:
                # This is a simplified approach - you may need to implement
                # more sophisticated membrane integration logic
                red_with_membrane = red_metamold.union(membrane)
                blue_with_membrane = blue_metamold.union(membrane)

                red_metamold = red_with_membrane
                blue_metamold = blue_with_membrane

            except Exception as e:
                print(f"Warning: Could not apply membrane {i}: {e}")
                continue

        # Save updated metamolds
        updated_red_path = os.path.join(results_dir, "metamold_red_with_membranes.stl")
        updated_blue_path = os.path.join(results_dir, "metamold_blue_with_membranes.stl")

        red_metamold.export(updated_red_path)
        blue_metamold.export(updated_blue_path)

        print(f"Updated metamolds saved with membranes")
        return updated_red_path, updated_blue_path

    def save_membranes(self, results_dir):
        """
        Save individual membrane files
        """
        if not self.membranes:
            return []

        membrane_paths = []
        for i, membrane in enumerate(self.membranes):
            membrane_path = os.path.join(results_dir, f"topological_membrane_{i + 1}.stl")
            membrane.export(membrane_path)
            membrane_paths.append(membrane_path)
            print(f"Saved membrane {i + 1} to {membrane_path}")

        return membrane_paths


def process_topological_membranes(mesh_path, metamold_red_path, metamold_blue_path, results_dir):
    """
    Main function to process topological membranes
    """
    print("Starting topological membrane processing...")

    # Initialize membrane processor
    membrane_processor = TopologicalMembranes(mesh_path)

    # Check if membranes are needed
    if membrane_processor.genus == 0:
        print("Mesh has genus 0, no topological membranes needed")
        return metamold_red_path, metamold_blue_path, []

    # Generate membranes
    membranes = membrane_processor.generate_membranes()

    if not membranes:
        print("No valid membranes could be generated")
        return metamold_red_path, metamold_blue_path, []

    # Save individual membranes
    membrane_paths = membrane_processor.save_membranes(results_dir)

    # Apply membranes to metamolds
    updated_red_path, updated_blue_path = membrane_processor.apply_membranes_to_metamolds(
        metamold_red_path, metamold_blue_path, results_dir
    )

    print(f"Topological membrane processing complete!")
    print(f"Generated {len(membranes)} membranes")

    return updated_red_path, updated_blue_path, membrane_paths


# # Usage example - add this to your main.py after metamold generation:
# if __name__ == "__main__":
#     # This would be added to your existing main.py after the metamold generation
#
#     # Process topological membranes
#     updated_metamold_red_path, updated_metamold_blue_path, membrane_paths = process_topological_membranes(
#         mesh_path, metamold_red_path, metamold_blue_path, results_dir
#     )
#
#     print("Final metamold paths with membranes:")
#     print(f"Red metamold: {updated_metamold_red_path}")
#     print(f"Blue metamold: {updated_metamold_blue_path}")
#     print(f"Membrane files: {membrane_paths}")