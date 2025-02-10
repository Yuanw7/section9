import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D

def surface1(x, y):
    return 2 * x**2 + 2 * y**2

def surface2(x, y):
    return 2 * np.exp(-x**2 - y**2)

# Generate grid of x and y values
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

# Compute z-values for both surfaces
Z_top = surface1(X, Y)
Z_bottom = surface2(X, Y)

# Create point clouds for top and bottom surfaces
points_top = np.column_stack([X.ravel(), Y.ravel(), Z_top.ravel()])
points_bottom = np.column_stack([X.ravel(), Y.ravel(), Z_bottom.ravel()])

# Combine points into a single array
points = np.vstack([points_top, points_bottom])
# Triangulate the top surface
tri_top = Delaunay(points_top[:, :2])
triangles_top = tri_top.simplices

# Triangulate the bottom surface and adjust indices
tri_bottom = Delaunay(points_bottom[:, :2])
triangles_bottom = tri_bottom.simplices + len(points_top)  # Offset indices

# Combine triangles from both surfaces
triangles = np.vstack([triangles_top, triangles_bottom])
# Plot the combined mesh
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(
    points[:, 0], 
    points[:, 1], 
    points[:, 2], 
    triangles=triangles, 
    cmap='viridis',
    edgecolor='none',
    alpha=0.7
)

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Combined Delaunay Triangulation of Top and Bottom Surfaces')
plt.show()
