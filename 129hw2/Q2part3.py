import numpy as np  # Add this line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parallel transport on sphere (from Problem 1)
def parallel_transport(theta0, phi_vals):
    # Transport along latitude (holonomy example)
    transported_vectors = []
    for phi in phi_vals:
        R = np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
        transported_vectors.append(R @ np.array([1, 0, 0]))
    return transported_vectors

# Closed loop (theta fixed, phi from 0 to 2π)
phi_vals = np.linspace(0, 2*np.pi, 50)
initial_vector = np.array([1, 0, 0])
transported = parallel_transport(np.pi/4, phi_vals)

# Project vectors
proj_points = [stereographic(np.array([np.sin(np.pi/4)*np.cos(phi), np.sin(np.pi/4)*np.sin(phi), np.cos(np.pi/4)])) for phi in phi_vals]
proj_vectors = [stereographic(v) - stereographic([0,0,0]) for v in transported]

# Plot
fig, ax = plt.subplots()
ax.quiver(*np.array(proj_points).T, *np.array(proj_vectors).T, scale=10, color='r')
ax.set_title("Parallel Transport Trajectories (Projected)")
plt.show()
