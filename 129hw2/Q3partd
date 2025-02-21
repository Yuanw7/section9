from mpl_toolkits.mplot3d import Axes3D

# Compute face normals
face_normals = []
for simplex in tri.simplices:
    p0, p1, p2 = points_3d[simplex]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    face_normals.append(normal)

# Plot normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='blue', s=10)

for i, simplex in enumerate(tri.simplices):
    centroid = np.mean(points_3d[simplex], axis=0)
    ax.quiver(centroid[0], centroid[1], centroid[2],
              face_normals[i][0], face_normals[i][1], face_normals[i][2],
              length=0.5, color='red')

ax.set_title("Surface Normals (Parabolic Lifting)")
plt.show()
