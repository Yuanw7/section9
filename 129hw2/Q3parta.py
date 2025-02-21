import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# Generate 2D point cloud
np.random.seed(0)
points = np.random.uniform(-2, 2, (50, 2))  # Points in [-2, 2]^2

# Compute structures
hull = ConvexHull(points)
tri = Delaunay(points)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], c='k', s=10, label='Points')
plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2, label='Convex Hull')
plt.triplot(points[:,0], points[:,1], tri.simplices, color='green', alpha=0.5, label='Delaunay Triangulation')
plt.legend()
plt.title("Convex Hull and Delaunay Triangulation")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
