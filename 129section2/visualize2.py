import numpy as np
import matplotlib.pyplot as plt
from convex_hull import graham_scan, jarvis_march, quickhull, monotone_chain

# Load data (skip header row if your mesh.dat has one)
points = np.loadtxt('mesh.dat', skiprows=1)  # Remove skiprows=1 if no header

# Compute convex hulls using all four methods
hull_graham = graham_scan(points)
hull_jarvis = jarvis_march(points)
hull_quick = quickhull(points)
hull_monotone = monotone_chain(points)

# Plot settings
plt.figure(figsize=(12, 8))
plt.scatter(points[:, 0], points[:, 1], s=10, c='black', alpha=0.5, label='Point Cloud')

# Helper function to plot closed hulls
def plot_hull(hull, color, label):
    if len(hull) < 2:
        return  # Skip invalid hulls
    hull_closed = np.vstack([hull, hull[0]])  # Close the hull loop
    plt.plot(hull_closed[:, 0], hull_closed[:, 1], color=color, linewidth=2, marker='o', markersize=5, label=label)

# Plot all hulls
plot_hull(hull_graham, 'red', 'Graham Scan')
plot_hull(hull_jarvis, 'green', 'Jarvis March')
plot_hull(hull_quick, 'blue', 'Quickhull')
plot_hull(hull_monotone, 'orange', 'Monotone Chain')

plt.legend()
plt.title('Convex Hull Algorithms Comparison')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.savefig('convex_hulls.png', dpi=300)
plt.close()
