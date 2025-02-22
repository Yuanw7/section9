import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Load mesh.dat
points = np.loadtxt('mesh.dat')
if points.shape[1] != 2:
    raise ValueError("mesh.dat must contain 2D coordinates (x, y)")

# Triangulate and lift to 3D
tri = Delaunay(points)
points_3d = np.column_stack((points[:,0], points[:,1], points[:,0]**2 + points[:,1]**2))

# Compute area ratios
area_ratios = []
for simplex in tri.simplices:
    tri_2d = points[simplex]
    tri_3d = points_3d[simplex]
    
    v1_2d = tri_2d[1] - tri_2d[0]
    v2_2d = tri_2d[2] - tri_2d[0]
    area_2d = 0.5 * np.abs(np.cross(v1_2d, v2_2d))
    
    v1_3d = tri_3d[1] - tri_3d[0]
    v2_3d = tri_3d[2] - tri_3d[0]
    area_3d = 0.5 * np.linalg.norm(np.cross(v1_3d, v2_3d))
    
    area_ratios.append(area_3d / area_2d)

# Plot using triplot (closer to original)
plt.figure(figsize=(10, 6))
plt.tripcolor(points[:,0], points[:,1], tri.simplices, facecolors=area_ratios, cmap='viridis')
plt.colorbar(label='Area Ratio (3D / 2D)')
plt.title("Area Ratio Heatmap (mesh.dat)")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
