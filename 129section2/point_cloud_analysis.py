import numpy as np
import time
import matplotlib.pyplot as plt


def generate_uniform_point_cloud(n, x_min=0, x_max=1, y_min=0, y_max=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(x_min, x_max, n)
    y = np.random.uniform(y_min, y_max, n)
    return np.column_stack((x, y))


