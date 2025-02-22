import numpy as np

# Avoid z=1 (north pole singularity)
P = np.array([0, 0, 0.9999])  # Near north pole
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

def stereographic(P, eps=1e-10):
    x, y, z = P
    scale = 1 / (1 - z + eps)
    return np.array([x * scale, y * scale])

# Project vectors
scale = 1 / (1 - P[2] + 1e-10)
proj_v1 = scale * v1[:2]
proj_v2 = scale * v2[:2]

# Compute inner products
ip_3d = np.dot(v1, v2)
ip_2d = np.dot(proj_v1, proj_v2)
conformal_factor = scale**2

print(f"Original Inner Product: {ip_3d}")
print(f"Projected Inner Product: {ip_2d:.2f} (Expected: {ip_3d * conformal_factor:.2f})")
