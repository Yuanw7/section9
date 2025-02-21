def compute_curvatures(x, y):
    # First fundamental form
    E = 1 + 4*x**2
    F = 4*x*y
    G = 1 + 4*y**2
    g = np.array([[E, F], [F, G]])
    
    # Second fundamental form
    L = 2 / np.sqrt(1 + 4*x**2 + 4*y**2)
    M = 0
    N = 2 / np.sqrt(1 + 4*x**2 + 4*y**2)
    II = np.array([[L, M], [M, N]])
    
    # Shape operator
    g_inv = np.linalg.inv(g)
    S = g_inv @ II
    
    # Principal curvatures
    k1, k2 = np.linalg.eigvals(S)
    K = k1 * k2  # Gaussian curvature
    H = (k1 + k2) / 2  # Mean curvature
    
    return K, H

# Compute curvatures
gaussian_curv = np.zeros(len(points))
mean_curv = np.zeros(len(points))
for i, (x, y) in enumerate(points):
    K, H = compute_curvatures(x, y)
    gaussian_curv[i] = K
    mean_curv[i] = H

# Plot Gaussian curvature
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c=gaussian_curv, cmap='viridis')
plt.colorbar(sc, label='Gaussian Curvature')
ax.set_title("Gaussian Curvature (Parabolic Lifting)")
plt.show()

