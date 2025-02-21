def lift_to_3d(points, f):
    return np.column_stack((points[:,0], points[:,1], f(points[:,0], points[:,1])))

# Lifting function z = x² + y²
f_parab = lambda x, y: x**2 + y**2
points_3d = lift_to_3d(points, f_parab)

# Compute area ratios
area_ratios = []
centroids = []
for simplex in tri.simplices:
    # 2D triangle area
    tri_2d = points[simplex]
    a = np.linalg.norm(tri_2d[1] - tri_2d[0])
    b = np.linalg.norm(tri_2d[2] - tri_2d[0])
    theta = np.arccos(np.dot(tri_2d[1]-tri_2d[0], tri_2d[2]-tri_2d[0]) / (a*b))
    area_2d = 0.5 * a * b * np.sin(theta)
    
    # 3D triangle area
    tri_3d = points_3d[simplex]
    v1 = tri_3d[1] - tri_3d[0]
    v2 = tri_3d[2] - tri_3d[0]
    area_3d = 0.5 * np.linalg.norm(np.cross(v1, v2))
    
    area_ratios.append(area_3d / area_2d)
    centroids.append(np.mean(tri_2d, axis=0))

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.scatter(np.array(centroids)[:,0], np.array(centroids)[:,1], c=area_ratios, cmap='viridis', s=100)
plt.colorbar(label='Area Ratio (3D / 2D)')
plt.title("Area Ratio Heatmap (Parabolic Lifting)")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
