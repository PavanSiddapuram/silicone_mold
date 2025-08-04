import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')


class STLMeshRepair:
    def __init__(self, stl_file_path):
        """
        Initialize the mesh repair tool with an STL file

        Args:
            stl_file_path (str): Path to the STL file
        """
        self.original_mesh = None
        self.repaired_mesh = None
        self.load_mesh(stl_file_path)

    def load_mesh(self, file_path):
        """Load STL file using trimesh"""
        try:
            self.original_mesh = trimesh.load_mesh(file_path)
            print(
                f"Mesh loaded successfully: {len(self.original_mesh.vertices)} vertices, {len(self.original_mesh.faces)} faces")
        except Exception as e:
            print(f"Error loading mesh: {e}")

    def analyze_mesh_issues(self):
        """Analyze mesh for common issues"""
        if self.original_mesh is None:
            print("No mesh loaded")
            return

        issues = {}

        # Check for watertight
        issues['is_watertight'] = self.original_mesh.is_watertight
        issues['is_winding_consistent'] = self.original_mesh.is_winding_consistent

        # Check for degenerate faces
        face_areas = self.original_mesh.area_faces
        degenerate_faces = np.sum(face_areas < 1e-10)
        issues['degenerate_faces'] = degenerate_faces

        # Check for duplicate vertices
        unique_vertices = np.unique(self.original_mesh.vertices, axis=0)
        issues['duplicate_vertices'] = len(self.original_mesh.vertices) - len(unique_vertices)

        # Check for holes/boundaries
        edges_unique = self.original_mesh.edges_unique
        edges_face_count = np.bincount(self.original_mesh.edges_unique_inverse)
        boundary_edges = edges_unique[edges_face_count == 1]
        issues['boundary_edges'] = len(boundary_edges)

        # Volume and surface area
        issues['volume'] = self.original_mesh.volume if self.original_mesh.is_watertight else "N/A (not watertight)"
        issues['surface_area'] = self.original_mesh.area

        return issues

    def print_analysis(self):
        """Print mesh analysis results"""
        issues = self.analyze_mesh_issues()

        print("\n=== MESH ANALYSIS ===")
        print(f"Watertight: {issues['is_watertight']}")
        print(f"Winding Consistent: {issues['is_winding_consistent']}")
        print(f"Degenerate Faces: {issues['degenerate_faces']}")
        print(f"Duplicate Vertices: {issues['duplicate_vertices']}")
        print(f"Boundary Edges: {issues['boundary_edges']}")
        print(f"Volume: {issues['volume']}")
        print(f"Surface Area: {issues['surface_area']:.4f}")

    def remove_duplicate_vertices(self, mesh):
        """Remove duplicate vertices"""
        vertices_rounded = np.round(mesh.vertices, decimals=6)
        unique_vertices, inverse_indices = np.unique(vertices_rounded, axis=0, return_inverse=True)

        # Update faces to use new vertex indices
        new_faces = inverse_indices[mesh.faces]

        # Create new mesh
        new_mesh = trimesh.Trimesh(vertices=unique_vertices, faces=new_faces)
        return new_mesh

    def remove_degenerate_faces(self, mesh):
        """Remove faces with zero or very small area"""
        face_areas = mesh.area_faces
        valid_faces = face_areas > 1e-10

        if np.sum(~valid_faces) > 0:
            print(f"Removing {np.sum(~valid_faces)} degenerate faces")
            new_faces = mesh.faces[valid_faces]
            new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces)
            return new_mesh
        return mesh

    def fix_winding_order(self, mesh):
        """
        Fix face winding order and ensure normals point outward.
        """
        try:
            mesh.fix_normals()
            if not mesh.is_winding_consistent:
                mesh = mesh.copy()
                mesh.invert()
            if mesh.volume < 0:
                print("Mesh appears inverted, flipping face winding...")
                mesh.invert()
            return mesh
        except Exception as e:
            print(f"Could not fix winding order: {e}")
            return mesh

    def fill_holes_poisson(self, mesh, depth=9):
        """Fill holes using Poisson surface reconstruction"""
        # Convert to Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Compute normals
        o3d_mesh.compute_vertex_normals()

        # Create point cloud from mesh
        pcd = o3d_mesh.sample_points_uniformly(number_of_points=50000)
        pcd.estimate_normals()

        # Poisson reconstruction
        try:
            poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, width=0, scale=1.1, linear_fit=False
            )

            # Convert back to trimesh
            vertices = np.asarray(poisson_mesh.vertices)
            faces = np.asarray(poisson_mesh.triangles)

            repaired_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return repaired_mesh
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")
            return mesh

    def wrap_repair(self):
        """Minimal repair: remove duplicates and degenerate faces only"""
        if self.original_mesh is None:
            print("No mesh loaded")
            return

        print("Performing minimal surface-preserving repair...")

        mesh = self.original_mesh.copy()
        mesh = self.remove_duplicate_vertices(mesh)
        mesh = self.remove_degenerate_faces(mesh)
        mesh = self.fix_winding_order(mesh)

        try:
            mesh.fill_holes()
        except:
            print("Could not fill small holes")

        return mesh

    def ball_pivoting_repair(self, radii=[0.005, 0.01, 0.02, 0.04]):
        """Alternative repair method using Ball Pivoting Algorithm"""
        if self.original_mesh is None:
            print("No mesh loaded")
            return

        # Sample points from mesh
        points = self.original_mesh.sample(30000)

        # Convert to Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        # Ball pivoting reconstruction
        radii_array = o3d.utility.DoubleVector(radii)
        try:
            bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, radii_array
            )

            # Convert to trimesh
            vertices = np.asarray(bpa_mesh.vertices)
            faces = np.asarray(bpa_mesh.triangles)

            return trimesh.Trimesh(vertices=vertices, faces=faces)
        except:
            print("Ball pivoting reconstruction failed")
            return self.original_mesh

    def basic_repair(self):
        """Basic mesh repair operations"""
        if self.original_mesh is None:
            print("No mesh loaded")
            return

        print("Performing basic repairs...")
        mesh = self.original_mesh.copy()

        # Remove duplicate vertices
        mesh = self.remove_duplicate_vertices(mesh)

        # Remove degenerate faces
        mesh = self.remove_degenerate_faces(mesh)

        # Fix winding order
        mesh = self.fix_winding_order(mesh)

        # Fill small holes using trimesh
        try:
            mesh.fill_holes()
        except:
            print("Could not fill holes with basic method")

        return mesh

    def repair_mesh(self, method='wrap', **kwargs):
        """
        Main repair function

        Args:
            method (str): 'wrap', 'basic', 'poisson', 'ball_pivoting'
            **kwargs: Additional parameters for specific methods
        """
        if method == 'wrap':
            self.repaired_mesh = self.wrap_repair(**kwargs)
        elif method == 'basic':
            self.repaired_mesh = self.basic_repair()
        elif method == 'poisson':
            self.repaired_mesh = self.fill_holes_poisson(self.original_mesh, **kwargs)
        elif method == 'ball_pivoting':
            self.repaired_mesh = self.ball_pivoting_repair(**kwargs)
        else:
            print(f"Unknown repair method: {method}")
            return

        if self.repaired_mesh is not None:
            print(f"Repair completed using {method} method")
            print(f"Repaired mesh: {len(self.repaired_mesh.vertices)} vertices, {len(self.repaired_mesh.faces)} faces")

    def compare_meshes(self):
        """Compare original and repaired mesh statistics"""
        if self.repaired_mesh is None:
            print("No repaired mesh available")
            return

        print("\n=== MESH COMPARISON ===")

        # Original mesh stats
        orig_issues = self.analyze_mesh_issues()

        # Repaired mesh stats
        temp_mesh = self.original_mesh
        self.original_mesh = self.repaired_mesh
        repair_issues = self.analyze_mesh_issues()
        self.original_mesh = temp_mesh

        print("ORIGINAL -> REPAIRED")
        print(f"Watertight: {orig_issues['is_watertight']} -> {repair_issues['is_watertight']}")
        print(f"Vertices: {len(self.original_mesh.vertices)} -> {len(self.repaired_mesh.vertices)}")
        print(f"Faces: {len(self.original_mesh.faces)} -> {len(self.repaired_mesh.faces)}")
        print(f"Boundary Edges: {orig_issues['boundary_edges']} -> {repair_issues['boundary_edges']}")
        print(f"Degenerate Faces: {orig_issues['degenerate_faces']} -> {repair_issues['degenerate_faces']}")

    def visualize_mesh(self, show_both=True, use_trimesh_viewer=True):
        """
        Visualize original and/or repaired mesh using Open3D or Trimesh viewer

        Args:
            show_both (bool): Show both original and repaired mesh
            use_trimesh_viewer (bool): Use trimesh viewer (True) or Open3D (False)
        """
        if use_trimesh_viewer:
            self._visualize_with_trimesh(show_both)
        else:
            self._visualize_with_open3d(show_both)

    def _visualize_with_trimesh(self, show_both=True):
        """Visualize using Trimesh's built-in viewer"""
        if show_both and self.repaired_mesh is not None:
            # Create scene with both meshes
            scene = trimesh.Scene()

            # Add original mesh (in red)
            original_colored = self.original_mesh.copy()
            original_colored.visual.face_colors = [255, 100, 100, 128]  # Semi-transparent red
            scene.add_geometry(original_colored, node_name='Original')

            # Add repaired mesh (in blue) - offset slightly for visibility
            repaired_colored = self.repaired_mesh.copy()
            repaired_colored.visual.face_colors = [100, 100, 255, 200]  # Semi-transparent blue
            # Offset repaired mesh slightly
            offset = np.array([self.original_mesh.bounds[1, 0] - self.original_mesh.bounds[0, 0] + 0.1, 0, 0])
            repaired_colored.vertices += offset
            scene.add_geometry(repaired_colored, node_name='Repaired')

            print("Showing both meshes: Original (red, left) and Repaired (blue, right)")
            scene.show()
        else:
            # Show single mesh
            mesh_to_show = self.repaired_mesh if self.repaired_mesh is not None else self.original_mesh
            title = "Repaired Mesh" if self.repaired_mesh is not None else "Original Mesh"
            print(f"Showing: {title}")
            mesh_to_show.show()

    def _visualize_with_open3d(self, show_both=True):
        """Visualize using Open3D viewer"""
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="STL Mesh Repair Viewer", width=1200, height=800)

        if show_both and self.repaired_mesh is not None:
            # Convert original mesh to Open3D
            o3d_original = o3d.geometry.TriangleMesh()
            o3d_original.vertices = o3d.utility.Vector3dVector(self.original_mesh.vertices)
            o3d_original.triangles = o3d.utility.Vector3iVector(self.original_mesh.faces)
            o3d_original.compute_vertex_normals()
            o3d_original.paint_uniform_color([1.0, 0.4, 0.4])  # Red for original

            # Convert repaired mesh to Open3D
            o3d_repaired = o3d.geometry.TriangleMesh()
            repaired_vertices = self.repaired_mesh.vertices.copy()
            # Offset repaired mesh for side-by-side viewing
            offset = np.array([self.original_mesh.bounds[1, 0] - self.original_mesh.bounds[0, 0] + 0.1, 0, 0])
            repaired_vertices += offset
            o3d_repaired.vertices = o3d.utility.Vector3dVector(repaired_vertices)
            o3d_repaired.triangles = o3d.utility.Vector3iVector(self.repaired_mesh.faces)
            o3d_repaired.compute_vertex_normals()
            o3d_repaired.paint_uniform_color([0.4, 0.4, 1.0])  # Blue for repaired

            vis.add_geometry(o3d_original)
            vis.add_geometry(o3d_repaired)
            print("Open3D Viewer: Original (red, left) and Repaired (blue, right)")
        else:
            # Show single mesh
            mesh_to_show = self.repaired_mesh if self.repaired_mesh is not None else self.original_mesh
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_to_show.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_to_show.faces)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray

            vis.add_geometry(o3d_mesh)
            title = "Repaired Mesh" if self.repaired_mesh is not None else "Original Mesh"
            print(f"Open3D Viewer: {title}")

        # Set render options
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.mesh_show_wireframe = False
        render_option.light_on = True

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def visualize_wireframe(self, mesh_type='both'):
        """
        Visualize mesh in wireframe mode using Open3D

        Args:
            mesh_type (str): 'original', 'repaired', or 'both'
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Wireframe View", width=1200, height=800)

        if mesh_type == 'both' and self.repaired_mesh is not None:
            # Original mesh wireframe
            o3d_original = o3d.geometry.TriangleMesh()
            o3d_original.vertices = o3d.utility.Vector3dVector(self.original_mesh.vertices)
            o3d_original.triangles = o3d.utility.Vector3iVector(self.original_mesh.faces)
            o3d_original.paint_uniform_color([1.0, 0.0, 0.0])  # Red

            # Repaired mesh wireframe
            o3d_repaired = o3d.geometry.TriangleMesh()
            repaired_vertices = self.repaired_mesh.vertices.copy()
            offset = np.array([self.original_mesh.bounds[1, 0] - self.original_mesh.bounds[0, 0] + 0.1, 0, 0])
            repaired_vertices += offset
            o3d_repaired.vertices = o3d.utility.Vector3dVector(repaired_vertices)
            o3d_repaired.triangles = o3d.utility.Vector3iVector(self.repaired_mesh.faces)
            o3d_repaired.paint_uniform_color([0.0, 0.0, 1.0])  # Blue

            vis.add_geometry(o3d_original)
            vis.add_geometry(o3d_repaired)
        else:
            if mesh_type == 'repaired' and self.repaired_mesh is not None:
                mesh_to_show = self.repaired_mesh
                color = [0.0, 0.0, 1.0]
            else:
                mesh_to_show = self.original_mesh
                color = [1.0, 0.0, 0.0]

            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_to_show.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_to_show.faces)
            o3d_mesh.paint_uniform_color(color)

            vis.add_geometry(o3d_mesh)

        # Enable wireframe mode
        render_option = vis.get_render_option()
        render_option.mesh_show_wireframe = True
        render_option.mesh_show_back_face = True
        render_option.line_width = 1.0

        print("Wireframe view - Use mouse to rotate, scroll to zoom")
        vis.run()
        vis.destroy_window()

    def visualize_point_cloud(self, num_points=10000):
        """Visualize mesh as point cloud using Open3D"""
        if self.original_mesh is None:
            print("No mesh loaded")
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud View", width=1200, height=800)

        # Sample points from original mesh
        points_orig = self.original_mesh.sample(num_points)
        pcd_orig = o3d.geometry.PointCloud()
        pcd_orig.points = o3d.utility.Vector3dVector(points_orig)
        pcd_orig.paint_uniform_color([1.0, 0.4, 0.4])  # Red

        vis.add_geometry(pcd_orig)

        if self.repaired_mesh is not None:
            # Sample points from repaired mesh
            points_repair = self.repaired_mesh.sample(num_points)
            pcd_repair = o3d.geometry.PointCloud()
            # Offset for side-by-side viewing
            offset = np.array([self.original_mesh.bounds[1, 0] - self.original_mesh.bounds[0, 0] + 0.1, 0, 0])
            points_repair += offset
            pcd_repair.points = o3d.utility.Vector3dVector(points_repair)
            pcd_repair.paint_uniform_color([0.4, 0.4, 1.0])  # Blue

            vis.add_geometry(pcd_repair)
            print("Point Cloud View: Original (red, left) and Repaired (blue, right)")
        else:
            print("Point Cloud View: Original mesh")

        vis.run()
        vis.destroy_window()

    def interactive_comparison(self):
        """Interactive comparison tool using Open3D"""
        if self.repaired_mesh is None:
            print("No repaired mesh available for comparison")
            self.visualize_mesh(show_both=False, use_trimesh_viewer=False)
            return

        print("\n=== INTERACTIVE MESH COMPARISON ===")
        print("1. Press '1' - Show original mesh only")
        print("2. Press '2' - Show repaired mesh only")
        print("3. Press '3' - Show both meshes")
        print("4. Press 'W' - Toggle wireframe mode")
        print("5. Press 'ESC' - Exit")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Interactive Mesh Comparison", width=1400, height=900)

        # Prepare meshes
        o3d_original = o3d.geometry.TriangleMesh()
        o3d_original.vertices = o3d.utility.Vector3dVector(self.original_mesh.vertices)
        o3d_original.triangles = o3d.utility.Vector3iVector(self.original_mesh.faces)
        o3d_original.compute_vertex_normals()
        o3d_original.paint_uniform_color([1.0, 0.4, 0.4])

        o3d_repaired = o3d.geometry.TriangleMesh()
        o3d_repaired.vertices = o3d.utility.Vector3dVector(self.repaired_mesh.vertices)
        o3d_repaired.triangles = o3d.utility.Vector3iVector(self.repaired_mesh.faces)
        o3d_repaired.compute_vertex_normals()
        o3d_repaired.paint_uniform_color([0.4, 0.4, 1.0])

        # Start with both meshes
        vis.add_geometry(o3d_original)
        vis.add_geometry(o3d_repaired)

        vis.run()
        vis.destroy_window()

    def export_mesh(self, filename, mesh_type='repaired'):
        """Export the mesh to file"""
        mesh_to_export = self.repaired_mesh if mesh_type == 'repaired' and self.repaired_mesh is not None else self.original_mesh

        if mesh_to_export is None:
            print("No mesh to export")
            return

        try:
            mesh_to_export.export(filename)
            print(f"Mesh exported to {filename}")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
