import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Load STL file
mesh = trimesh.load("results/cow/metamold_blue_wrapped.stl")

# Get face centroids and normals
face_centroids = mesh.triangles_center
face_normals = mesh.face_normals

# Find base face (lowest Z)
base_face_idx = np.argmin(face_centroids[:, 2])
base_centroid = face_centroids[base_face_idx]
base_normal_outward = face_normals[base_face_idx]
base_normal_inward = -base_normal_outward  # Flip to point into the STL

# Get all vertices for plotting
vertices = mesh.vertices
faces = mesh.faces

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot mesh
ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor='lightgrey', edgecolor='black', alpha=0.5))

# Plot the base face centroid
ax.scatter(*base_centroid, color='red', s=50, label='Base Centroid')

# Plot the normal (as an arrow)
arrow_length = 10
ax.quiver(*base_centroid,
          *base_normal_inward * arrow_length,
          color='blue', linewidth=2, label='Base Normal (inward)')

# Set axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Base Face with Inward Normal Vector")
ax.legend()
ax.auto_scale_xyz(*np.transpose(vertices))

plt.tight_layout()
plt.show()
