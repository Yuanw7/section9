import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------
# Convex Hull Algorithm
# ---------------------------

def graham_scan(points):
    """Computes the convex hull of a set of 2D points using Graham's scan algorithm."""
    print(f"🔹 Starting Graham's scan with {len(points)} points...")
    points = sorted(points, key=lambda p: (p[0], p[1]))  # Sort by x, then y
    print(f"🔹 Points sorted: {points[:5]}...")  # Show first 5 points

    lower, upper = [], []
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    print(f"✅ Convex hull computed with {len(lower[:-1] + upper[:-1])} points.")
    return np.array(lower[:-1] + upper[:-1])

# ---------------------------
# Visualization Function
# ---------------------------

def plot_hulls(points, hull, filename="convex_hulls.png"):
    """Plots the original points and the convex hull."""
    print("🔹 Plotting convex hull...")
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=10, label='Original Points')
    
    if len(hull) > 0:
        closed_hull = np.vstack([hull, hull[0]])  # Close the hull polygon
        plt.plot(closed_hull[:, 0], closed_hull[:, 1], 'r-', linewidth=2, marker='o', markersize=5, label='Convex Hull')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Convex Hull Visualization')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(filename)  # Save the figure
    print(f"✅ Plot saved as '{filename}'.")

# ---------------------------
# Main Processing Function
# ---------------------------

def process_and_visualize(filename):
    """Loads data, computes the convex hull, and visualizes it."""
    try:
        print("🔹 Checking if the file exists...")
        if not os.path.exists(filename):
            print(f"❌ Error: File '{filename}' not found.")
            return None

        print(f"🔹 Loading file: {filename}")
        points = np.loadtxt(filename, skiprows=1)  # Skip header if present

        print("🔹 Checking file format...")
        if points.shape[1] != 2:
            raise ValueError("❌ Error: File should contain 2D coordinates")

        print(f"✅ Processing {len(points)} points...")

        hull = graham_scan(points)
        plot_hulls(points, hull, filename="convex_hulls.png")

        print("✅ Visualization completed.")

        return hull
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ---------------------------
# Execute Program
# ---------------------------

if __name__ == "__main__":
    mesh_file_path = "/root/Desktop/yambo/mesh.dat"  # Adjusted path for Docker
    process_and_visualize(mesh_file_path)
