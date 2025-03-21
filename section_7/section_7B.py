import numpy as np
import matplotlib.pyplot as plt

# Direction numbers for the second dimension (given in the problem)
m_values = [1, 3, 5, 7, 1, 3]  # m1=1, m2=3, m3=5, m4=7, m5=1, m6=3
max_bits = len(m_values)
dim2_dir = [m / (2 ** (i+1)) for i, m in enumerate(m_values)]  # v_{2,j} = m_j / 2^j

# Direction numbers for the first dimension (default: v_{1,j} = 1/2^j)
dim1_dir = [1 / (2 ** (i+1)) for i in range(max_bits)]

def sobol_coordinate(n, direction_numbers):
    """
    Compute the Sobol coordinate for index `n` using given direction numbers.
    """
    n_binary = bin(n)[2:][::-1]  # Reverse binary to start from LSB
    coordinate = 0.0
    for i in range(len(n_binary)):
        if n_binary[i] == '1':
            coordinate += direction_numbers[i]
    return coordinate % 1  # Ensure value is in [0, 1)

# Generate first 50 points for both dimensions
points = []
for n in range(1, 51):  # Indices 1 to 50
    x = sobol_coordinate(n, dim1_dir)
    y = sobol_coordinate(n, dim2_dir)
    points.append((x, y))

# Plot the 2D Sobol sequence
x_coords, y_coords = zip(*points)
plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, c='blue', alpha=0.6)
plt.title("2D Sobol Sequence (First 50 Points)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
