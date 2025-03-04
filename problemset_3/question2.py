import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --------------- Configuration ---------------
BASE_DIR = 'Local density of states near band edge'
HEATMAP_DIR = os.path.join(BASE_DIR, 'local density of states heatmap')
HEIGHT_DIR = os.path.join(BASE_DIR, 'local density of states height')

# Define sub-region for Part (c) (adjust based on data dimensions)
SUB_REGION = {
    "x_start": 50,  # Column index start
    "x_end": 150,   # Column index end
    "y_start": 50,  # Row index start
    "y_end": 150    # Row index end
}

# Colormaps for plots
HEATMAP_CMAP = 'inferno'
SURFACE_CMAP = 'viridis'

# ----------------------------------------------

def validate_subregion(data_shape):
    """Ensure sub-region is within data bounds."""
    valid = (
        SUB_REGION["x_start"] >= 0 and
        SUB_REGION["x_end"] <= data_shape[1] and
        SUB_REGION["y_start"] >= 0 and
        SUB_REGION["y_end"] <= data_shape[0]
    )
    if not valid:
        raise ValueError("Sub-region exceeds data dimensions. Adjust SUB_REGION.")
    return

def process_ldos_files():
    # Create output directories
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(HEIGHT_DIR, exist_ok=True)

    # Track sub-region averages for Part (c)
    subregion_avg = []
    levels = list(range(11))  # Levels 0-10

    for level in levels:
        file_name = f'local_density_of_states_for_level_{level}.txt'
        file_path = os.path.join(BASE_DIR, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

        # Load data (assuming 230x230 grid)
        data = np.loadtxt(file_path)

        # --------------------- Part (a): 2D Heatmap ---------------------
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap=HEATMAP_CMAP, origin='lower')
        plt.colorbar(label='LDOS Intensity')
        plt.title(f'LDOS Heatmap - Level {level}')
        plt.axis('off')
        heatmap_save_path = os.path.join(HEATMAP_DIR, f'level_{level}_heatmap.png')
        plt.savefig(heatmap_save_path, bbox_inches='tight', dpi=150)
        plt.close()

        # --------------------- Part (b): 3D Surface Plot ---------------------
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(
            X, Y, data, cmap=SURFACE_CMAP,
            rstride=5, cstride=5, linewidth=0.1, antialiased=True
        )
        fig.colorbar(surf, ax=ax, label='LDOS Intensity', shrink=0.5)
        ax.set_title(f'LDOS Surface Plot - Level {level}')
        surface_save_path = os.path.join(HEIGHT_DIR, f'level_{level}_surface.png')
        plt.savefig(surface_save_path, bbox_inches='tight', dpi=150)
        plt.close()

        # --------------------- Part (c): Sub-region Analysis ---------------------
        validate_subregion(data.shape)
        sub_data = data[
            SUB_REGION["y_start"]:SUB_REGION["y_end"],
            SUB_REGION["x_start"]:SUB_REGION["x_end"]
        ]
        subregion_avg.append(np.mean(sub_data))

    # Plot sub-region analysis
    plt.figure(figsize=(8, 5))
    plt.plot(levels, subregion_avg, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Energy Level Index', fontsize=12)
    plt.ylabel('Average LDOS', fontsize=12)
    plt.title(f'Average LDOS in Sub-region ({SUB_REGION["x_start"]}:{SUB_REGION["x_end"]}, ' +
              f'{SUB_REGION["y_start"]}:{SUB_REGION["y_end"]})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(levels)
    plt.tight_layout()
    plot_save_path = os.path.join(BASE_DIR, 'subregion_average_analysis.png')
    plt.savefig(plot_save_path, dpi=150)
    plt.close()

    print("All tasks completed successfully!")

if __name__ == '__main__':
    process_ldos_files()
