import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# Load data from mesh.dat
points = np.loadtxt('mesh.dat')  # Ensure file is in the same directory

# Check data shape
print("Data shape:", points.shape)  # Debug step

# Compute Convex Hull
hull = ConvexHull(points)

# Compute Delaunay Triangulation
tri = Delaunay(points)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], c='b', s=10, label='Points')
plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2, label='Convex Hull')
plt.triplot(points[:,0], points[:,1], tri.simplices, color='green', alpha=0.5, label='Delaunay Triangulation')
plt.legend()
plt.title("Convex Hull and Delaunay Triangulation")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
