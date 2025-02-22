import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Stereographic projection with numerical stability
def stereographic(P, eps=1e-10):
    x, y, z = P
    scale = 1 / (1 - z + eps)  # Add epsilon to avoid division by zero
    return np.array([x * scale, y * scale])

# Generate curves avoiding z=1
theta = np.linspace(0.01, 2*np.pi - 0.01, 100)  # Exclude θ=0 and θ=2π
curve1 = np.array([np.sin(theta), np.zeros_like(theta), np.cos(theta)])  # Vertical circle
curve2 = np.array([np.zeros_like(theta), np.sin(theta), np.cos(theta)])  # Horizontal circle

# Project curves
proj_curve1 = np.array([stereographic(curve1[:,i]) for i in range(len(theta))]).T
proj_curve2 = np.array([stereographic(curve2[:,i]) for i in range(len(theta))]).T

# Compute tangents at intersection (theta=pi/4)
idx = np.argmin(np.abs(theta - np.pi/4))  # Choose a point away from z=1
tangent1 = np.array([np.cos(theta[idx]), 0, -np.sin(theta[idx])])
tangent2 = np.array([0, np.cos(theta[idx]), -np.sin(theta[idx])])

# Project tangents using Jacobian
z_val = curve1[2, idx]
scale = 1 / (1 - z_val + 1e-10)  # Avoid division by zero
J = scale * np.eye(2)  # Jacobian is scalar * identity
proj_tangent1 = J @ tangent1[:2]
proj_tangent2 = J @ tangent2[:2]

# Compute angles
angle_3d = np.arccos(np.dot(tangent1, tangent2))
angle_2d = np.arccos(np.dot(proj_tangent1, proj_tangent2) / (np.linalg.norm(proj_tangent1) * np.linalg.norm(proj_tangent2)))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(*curve1, label='Curve 1')
ax1.plot(*curve2, label='Curve 2')
ax1.set_title("Original Curves on Sphere")

ax2.plot(*proj_curve1, label='Projected Curve 1')
ax2.plot(*proj_curve2, label='Projected Curve 2')
ax2.annotate(f'Original Angle: {np.degrees(angle_3d):.2f}°', (0.1, 0.9), xycoords='axes fraction')
ax2.annotate(f'Projected Angle: {np.degrees(angle_2d):.2f}°', (0.1, 0.85), xycoords='axes fraction')
ax2.set_title("Projected Curves (Conformal)")
ax2.legend()
plt.show()
