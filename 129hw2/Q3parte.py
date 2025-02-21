vertex_normals = np.zeros_like(points_3d)
vertex_counts = np.zeros(len(points_3d))

for i, simplex in enumerate(tri.simplices):
    for idx in simplex:
        vertex_normals[idx] += face_normals[i]
        vertex_counts[idx] += 1

vertex_normals /= vertex_counts[:, None]

# Plot vertex normals
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='blue', s=10)

for i in range(len(points_3d)):
    ax.quiver(points_3d[i,0], points_3d[i,1], points_3d[i,2],
              vertex_normals[i,0], vertex_normals[i,1], vertex_normals[i,2],
              length=0.3, color='green')

ax.set_title("Vertex Normals (Parabolic Lifting)")
plt.show()
