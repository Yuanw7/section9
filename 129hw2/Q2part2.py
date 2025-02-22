import numpy as np  # Add this line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametrize great circles
phi = np.linspace(0, 2*np.pi, 100)  # Now works!
equator = np.array([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
meridian = np.array([np.zeros_like(phi), np.sin(phi), np.cos(phi)])
tilted = np.array([np.cos(phi)*np.cos(np.pi/4), np.sin(phi), np.cos(phi)*np.sin(np.pi/4)])

# Stereographic projection function
def stereographic(P):
    x, y, z = P
    scale = 1 / (1 - z + 1e-10)  # Avoid division by zero
    return np.array([x * scale, y * scale])

# Project curves
proj_eq = stereographic(equator)
proj_mer = stereographic(meridian)
proj_tilt = stereographic(tilted)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(*equator, label='Equator')
ax1.plot(*meridian, label='Meridian')
ax1.plot(*tilted, label='Tilted')
ax1.set_title("Great Circles on Sphere")

ax2.plot(*proj_eq, label='Equator (Circle)')
ax2.plot(*proj_mer[0], proj_mer[1], label='Meridian (Line)')
ax2.plot(*proj_tilt, label='Tilted (Circle)')
ax2.set_title("Projected Great Circles")
ax2.legend()
plt.show()
