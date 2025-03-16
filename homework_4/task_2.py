# part a
import numpy as np
import matplotlib.pyplot as plt

def generate_julia(c, xmin=-1.5, xmax=1.5, ymin=-1, ymax=1, resolution=800, max_iter=256):
    # Create grid of complex numbers
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Initialize iteration count and output array
    iterations = np.zeros(Z.shape, dtype=int)
    julia_mask = np.full(Z.shape, True, dtype=bool)  # Track bounded points
    
    for i in range(max_iter):
        Z[julia_mask] = Z[julia_mask] ** 2 + c
        mask = (np.abs(Z) < 2) & julia_mask  # Update mask for next iteration
        iterations += mask  # Increment count for escaping points
        julia_mask = mask
    
    return iterations

# Parameters
c = -0.7 + 0.356j
xmin, xmax, ymin, ymax = -1.5, 1.5, -1, 1
resolution = 800
max_iter = 256

# Generate Julia set
julia = generate_julia(c, xmin, xmax, ymin, ymax, resolution, max_iter)

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap='twilight_shifted', origin='lower')
plt.colorbar(label='Iterations to Escape')
plt.title(f'Julia Set for c = {c}')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.show()

# part b

from scipy.spatial import ConvexHull

# Extract coordinates of points in the Julia set
y_indices, x_indices = np.where(julia < max_iter)  # Points that did not escape
x_coords = np.linspace(xmin, xmax, resolution)[x_indices]
y_coords = np.linspace(ymin, ymax, resolution)[y_indices]
points = np.column_stack((x_coords, y_coords))

# Compute convex hull
hull = ConvexHull(points)
hull_area = hull.volume  # For 2D, volume is the area

print(f"Convex Hull Area: {hull_area:.4f}")

# part c
from skimage.measure import find_contours

# Binarize the Julia set (1 for points in the set, 0 otherwise)
julia_binary = (julia < max_iter).astype(int)

# Find contours at a level of 0.5 (boundary)
contours = find_contours(julia_binary, 0.5)

# Convert contour coordinates to real coordinates
def scale_contour(contour, xmin, xmax, ymin, ymax, resolution):
    x = xmin + (contour[:, 1] / resolution) * (xmax - xmin)
    y = ymin + (contour[:, 0] / resolution) * (ymax - ymin)
    return np.column_stack((x, y))

# Compute area using the largest contour
if len(contours) > 0:
    largest_contour = max(contours, key=lambda x: len(x))
    scaled_contour = scale_contour(largest_contour, xmin, xmax, ymin, ymax, resolution)
    
    # Calculate area using the shoelace formula
    def polygon_area(vertices):
        x, y = vertices.T
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    contour_area = polygon_area(scaled_contour)
    print(f"Contour Area: {contour_area:.4f}")
else:
    print("No contour found.")

# part d

def box_counting(binary_image, box_sizes):
    N = []
    for size in box_sizes:
        # Split image into boxes of size x size
        grid = binary_image.shape[0] // size
        count = 0
        
        for i in range(0, binary_image.shape[0], size):
            for j in range(0, binary_image.shape[1], size):
                # Check if any pixel in the box is part of the set
                if np.any(binary_image[i:i+size, j:j+size]):
                    count += 1
        N.append(count)
    return np.array(N)

# Binarize the Julia set
binary_julia = (julia < max_iter).astype(int)

# Define box sizes (powers of 2 for simplicity)
box_sizes = 2 ** np.arange(1, 8)  # e.g., 2, 4, 8, ..., 128
epsilon = 1 / box_sizes  # Relative to image size

# Compute N(epsilon)
N = box_counting(binary_julia, box_sizes)

# Linear regression for log(N) vs log(1/epsilon)
A = np.vstack([np.log(1/epsilon), np.ones(len(epsilon))]).T
slope, intercept = np.linalg.lstsq(A, np.log(N), rcond=None)[0]
D = slope

print(f"Fractal Dimension (Box-Counting): D = {D:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(np.log(1/epsilon), np.log(N), 'bo-', label='Data')
plt.plot(np.log(1/epsilon), slope * np.log(1/epsilon) + intercept, 'r--', 
         label=f'Fit: D = {D:.4f}')
plt.xlabel('log(1/ε)')
plt.ylabel('log(N(ε))')
plt.title('Box-Counting Fractal Dimension')
plt.legend()
plt.show()
