import numpy as np
import matplotlib.pyplot as plt

points = np.loadtxt('mesh.dat', skiprows=1)


# Visualize
plt.scatter(points[:, 0], points[:, 1], s=5, c='blue', label='Point Cloud')
plt.legend()
plt.title('Raw Point Cloud')
plt.savefig('raw_pointcloud.png')
plt.close()
