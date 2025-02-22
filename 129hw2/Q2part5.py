import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Parallel transport around latitude (θ=π/4)
initial_vector = np.array([1, 0, 0])
final_vector = parallel_transport(np.pi/4, [2*np.pi])[-1]

# Compute holonomy angle
rotation_angle = np.arccos(np.dot(initial_vector, final_vector))

print(f"HOLONOMY ANGLE (Sphere): {np.degrees(rotation_angle):.2f}°")

# Project vectors
proj_initial = stereographic(initial_vector)
proj_final = stereographic(final_vector)
proj_angle = np.arccos(np.dot(proj_initial, proj_final) / (np.linalg.norm(proj_initial)*np.linalg.norm(proj_final)))

print(f"Projected Angle: {np.degrees(proj_angle):.2f}° (Same due to conformality)")
