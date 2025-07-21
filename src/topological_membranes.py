import trimesh
import numpy as np
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
        For a closed manifold: χ = V - E + F = 2 - 2g (where g is genus)
        So: genus = (2 - χ) / 2
        """
        # First check if mesh is watertight (closed)
        if not self.mesh.is_watertight:
            print("Warning: Mesh is not watertight, genus calculation may be inaccurate")

        V = len(self.mesh.vertices)
        F = len(self.mesh.faces)

        # For triangle mesh: E = 3F/2 (each face has 3 edges, each edge shared by 2 faces)
        # But let's use the actual edge count from trimesh
        try:
            E = len(self.mesh.edges)
        except:
            # Fallback: estimate edges for triangle mesh
            E = len(self.mesh.edges_unique)

        euler_char = V - E + F

        # For closed orientable surface: χ = 2 - 2g
        # So: g = (2 - χ) / 2
        genus = (2 - euler_char) / 2

        print(f"Mesh stats: V={V}, E={E}, F={F}")
        print(f"Euler characteristic: {euler_char}")
        print(f"Calculated genus: {genus}")

        # Genus should be non-negative integer
        genus = max(0, int(round(genus)))

        # Sanity check: if genus is still very large, likely calculation error
        if genus > 50:
            print(f"Warning: Calculated genus ({genus}) seems too high, likely mesh has issues")
            print("Checking mesh properties...")
            print(f"  Is watertight: {self.mesh.is_watertight}")
            print(f"  Is winding consistent: {self.mesh.is_winding_consistent}")
            print(f"  Volume: {self.mesh.volume}")

            # For problematic meshes, use a more conservative approach
            if not self.mesh.is_watertight:
                genus = 0  # Assume genus 0 for non-watertight meshes
                print("Setting genus to 0 due to non-watertight mesh")
            else:
                # Cap at reasonable value
                genus = min(genus, 5)
                print(f"Capping genus at {genus} for safety")

        print(f"Final genus: {genus}")
        return genus

    def find_tunnel_loops_simple(self):
        """
        Simplified tunnel detection using mesh boundaries and holes
        """
        if self.genus == 0:
            print("Genus 0: No tunnels detected")
            return []

        print(f"Genus {self.genus}: Looking for tunnel loops...")

        # Simple approach: find boundary edges that might form tunnel loops
        boundary_edges = []
        edge_count = {}

        # Count how many faces share each edge
        for face in self.mesh.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

        # Find edges that are only in one face (boundary edges)
        for edge, count in edge_count.items():
            if count == 1:
                boundary_edges.append(edge)

        print(f"Found {len(boundary_edges)} boundary edges")

        # Group boundary edges into loops
        if len(boundary_edges) == 0:
            # No boundary edges - try alternative approach
            # Sample some vertex loops from the mesh structure
            return self.find_sample_loops()

        # Build adjacency for boundary vertices
        boundary_graph = {}
        for edge in boundary_edges:
            v1, v2 = edge
            if v1 not in boundary_graph:
                boundary_graph[v1] = []
            if v2 not in boundary_graph:
                boundary_graph[v2] = []
            boundary_graph[v1].append(v2)
            boundary_graph[v2].append(v1)

        # Find connected boundary components
        visited = set()
        loops = []

        for start_vertex in boundary_graph:
            if start_vertex not in visited:
                loop = self.trace_boundary_loop(start_vertex, boundary_graph, visited)
                if len(loop) >= 4:  # Minimum loop size
                    loops.append(loop)

        # Limit to reasonable number of loops
        self.tunnel_loops = loops[:min(self.genus, len(loops))]
        print(f"Found {len(self.tunnel_loops)} tunnel loop candidates")
        return self.tunnel_loops

    def find_sample_loops(self):
        """
        Alternative method: find loops by sampling mesh vertices
        """
        print("Using alternative loop detection...")

        # Use mesh edges to build vertex graph
        vertex_graph = {}
        for edge in self.mesh.edges:
            v1, v2 = edge
            if v1 not in vertex_graph:
                vertex_graph[v1] = set()
            if v2 not in vertex_graph:
                vertex_graph[v2] = set()
            vertex_graph[v1].add(v2)
            vertex_graph[v2].add(v1)

        # Find some circular paths
        loops = []
        tried_starts = set()

        # Sample starting vertices
        sample_size = min(20, len(self.mesh.vertices))
        sample_vertices = np.random.choice(len(self.mesh.vertices), sample_size, replace=False)

        for start_v in sample_vertices:
            if start_v in tried_starts:
                continue
            tried_starts.add(start_v)

            loop = self.find_loop_from_vertex(start_v, vertex_graph, max_length=15)
            if loop and len(loop) >= 5:
                loops.append(loop)
                if len(loops) >= self.genus:
                    break

        return loops[:self.genus]

    def find_loop_from_vertex(self, start, vertex_graph, max_length=15):
        """
        Find a loop starting from a vertex using DFS
        """
        if start not in vertex_graph:
            return None

        def dfs(current, path, remaining_depth):
            if remaining_depth <= 0:
                return None

            if len(path) >= 4 and current == start:
                return path[:]

            if current in vertex_graph:
                neighbors = list(vertex_graph[current])
                np.random.shuffle(neighbors)  # Random order

                for neighbor in neighbors[:3]:  # Limit branching
                    if neighbor == start and len(path) >= 4:
                        return path + [neighbor]
                    elif neighbor not in path:
                        result = dfs(neighbor, path + [neighbor], remaining_depth - 1)
                        if result:
                            return result
            return None

        return dfs(start, [start], max_length)

    def trace_boundary_loop(self, start_vertex, boundary_graph, visited):
        """
        Trace a boundary loop starting from a vertex
        """
        loop = [start_vertex]
        current = start_vertex
        visited.add(current)

        while len(loop) < 50:  # Prevent infinite loops
            neighbors = [v for v in boundary_graph.get(current, []) if v not in visited or v == start_vertex]

            if not neighbors:
                break

            if start_vertex in neighbors and len(loop) > 3:
                # Found complete loop
                break

            # Choose next vertex (prefer unvisited)
            next_vertex = neighbors[0]
            if next_vertex == start_vertex:
                break

            loop.append(next_vertex)
            visited.add(next_vertex)
            current = next_vertex

        return loop

    def create_membrane_surface(self, loop_vertices, membrane_thickness=0.002):
        """
        Create a simple membrane surface for a loop
        """
        if len(loop_vertices) < 3:
            return None

        loop_points = self.mesh.vertices[loop_vertices]
        centroid = np.mean(loop_points, axis=0)

        # Simple planar membrane
        vertices = [centroid]  # Center point
        vertices.extend(loop_points)  # Loop points

        vertices = np.array(vertices)

        # Create triangular faces from center to loop
        faces = []
        n_loop = len(loop_points)
        for i in range(n_loop):
            next_i = (i + 1) % n_loop
            faces.append([0, i + 1, next_i + 1])

        # Create solid membrane by duplicating with offset
        # Estimate normal direction
        if n_loop >= 3:
            v1 = loop_points[1] - loop_points[0]
            v2 = loop_points[2] - loop_points[0]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)
            else:
                normal = np.array([0, 0, 1])
        else:
            normal = np.array([0, 0, 1])

        # Offset vertices
        offset_vertices = vertices + normal * membrane_thickness
        all_vertices = np.vstack([vertices, offset_vertices])

        # Original and flipped faces
        all_faces = faces[:]
        offset_faces = [[f[0] + len(vertices), f[2] + len(vertices), f[1] + len(vertices)] for f in faces]
        all_faces.extend(offset_faces)

        try:
            membrane = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
            membrane.fix_normals()
            return membrane
        except Exception as e:
            print(f"Warning: Failed to create membrane: {e}")
            return None

    def generate_membranes(self):
        """
        Generate membranes for detected tunnels
        """
        if self.genus == 0:
            print("No membranes needed (genus 0)")
            return []

        tunnel_loops = self.find_tunnel_loops_simple()
        membranes = []

        for i, loop in enumerate(tunnel_loops):
            print(f"Creating membrane {i + 1} for loop with {len(loop)} vertices...")
            membrane = self.create_membrane_surface(loop)
            if membrane is not None:
                membranes.append(membrane)
                print(f"  ✓ Membrane {i + 1}: {len(membrane.vertices)} vertices, {len(membrane.faces)} faces")
            else:
                print(f"  ✗ Failed to create membrane {i + 1}")

        self.membranes = membranes
        return membranes

    def save_membranes(self, results_dir):
        """
        Save membrane files
        """
        if not self.membranes:
            return []

        membrane_paths = []
        for i, membrane in enumerate(self.membranes):
            membrane_path = os.path.join(results_dir, f"membrane_{i + 1}.stl")
            try:
                membrane.export(membrane_path)
                membrane_paths.append(membrane_path)
                print(f"✓ Saved: {membrane_path}")
            except Exception as e:
                print(f"✗ Failed to save membrane {i + 1}: {e}")

        return membrane_paths

    def simple_visualization(self, results_dir):
        """
        Simple visualization using trimesh's built-in viewer
        """
        if not self.membranes:
            print("No membranes to visualize")
            return

        print("\nDisplaying membranes...")


def integrate_membranes_with_metamolds(red_path, blue_path, membranes, results_dir):
    """
    Integrate membrane surfaces with existing metamolds
    """
    try:
        # Load existing metamolds
        red_metamold = trimesh.load(red_path)
        blue_metamold = trimesh.load(blue_path)

        print(f"Integrating {len(membranes)} membranes with metamolds...")

        # Create combined meshes
        red_components = [red_metamold]
        blue_components = [blue_metamold]

        # Add membranes to both red and blue metamolds
        # (membranes will be printed in both colors for structural integrity)
        for i, membrane in enumerate(membranes):
            if membrane is not None and membrane.is_valid:
                red_components.append(membrane)
                blue_components.append(membrane)
                print(f"  ✓ Added membrane {i + 1} to both metamolds")
            else:
                print(f"  ✗ Skipped invalid membrane {i + 1}")

        # Combine meshes
        if len(red_components) > 1:
            final_red = trimesh.util.concatenate(red_components)
            final_red_path = os.path.join(results_dir, "final_metamold_red_with_membranes.stl")
            final_red.export(final_red_path)
            print(f"✓ Saved final red metamold: {final_red_path}")
        else:
            final_red_path = red_path
            print("No membranes added to red metamold")

        if len(blue_components) > 1:
            final_blue = trimesh.util.concatenate(blue_components)
            final_blue_path = os.path.join(results_dir, "final_metamold_blue_with_membranes.stl")
            final_blue.export(final_blue_path)
            print(f"✓ Saved final blue metamold: {final_blue_path}")
        else:
            final_blue_path = blue_path
            print("No membranes added to blue metamold")

        return final_red_path, final_blue_path

    except Exception as e:
        print(f"Error integrating membranes: {e}")
        return red_path, blue_path


def process_topological_membranes(mesh_path, metamold_red_path, metamold_blue_path, results_dir):
    """
    Process topological membranes using the TopologicalMembranes class
    """
    try:
        # Initialize the topological membrane processor
        topo_processor = TopologicalMembranes(mesh_path)

        print(f"Initialized topological processor for mesh: {mesh_path}")
        print(f"Detected genus: {topo_processor.genus}")

        # Generate membranes if genus > 0
        if topo_processor.genus > 0:
            print(f"\nGenerating membranes for genus {topo_processor.genus} object...")

            # Generate the membrane surfaces
            membranes = topo_processor.generate_membranes()

            if membranes:
                # Save individual membrane files
                membrane_paths = topo_processor.save_membranes(results_dir)

                # Integrate membranes with existing metamolds
                final_red_path, final_blue_path = integrate_membranes_with_metamolds(
                    metamold_red_path, metamold_blue_path, membranes, results_dir
                )

                return final_red_path, final_blue_path, membrane_paths
            else:
                print("No membranes could be generated")
                return metamold_red_path, metamold_blue_path, []
        else:
            print("Genus 0 object - no topological membranes needed")
            return metamold_red_path, metamold_blue_path, []

    except Exception as e:
        print(f"Error in topological membrane processing: {e}")
        print("Continuing with original metamolds...")
        return metamold_red_path, metamold_blue_path, []