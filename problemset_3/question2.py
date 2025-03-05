import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ----------------- Configuration -----------------
ROOT_DIR = os.getcwd()
DATA_DIR = ROOT_DIR
HEATMAP_DIR = os.path.join(ROOT_DIR, "heatmaps")
HEIGHT_DIR = os.path.join(ROOT_DIR, "height_plots")

SUB_REGION = {"x_start": 50, "x_end": 150, "y_start": 50, "y_end": 150}
# -------------------------------------------------

def process_files():
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(HEIGHT_DIR, exist_ok=True)
    subregion_avg = []
    
    for level in range(11):
        try:
            filename = f'local_density_of_states_for_level_{level}.txt'
            filepath = os.path.join(DATA_DIR, filename)
            
            # Load data while stripping commas
            data = np.loadtxt(
                filepath,
                delimiter=',',
                converters={i: lambda x: float(x.decode().strip().rstrip(',')) for i in range(230)}
            )
            
            # Part (a): Heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(data, cmap='inferno', origin='lower')
            plt.colorbar(label='LDOS Intensity')
            plt.title(f'Level {level} LDOS Heatmap')
            plt.axis('off')
            plt.savefig(os.path.join(HEATMAP_DIR, f'level_{level}_heatmap.png'), dpi=150)
            plt.close()
            
            # Part (b): 3D Plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
            ax.plot_surface(X, Y, data, cmap='viridis', rstride=5, cstride=5)
            plt.title(f'Level {level} LDOS Surface')
            plt.savefig(os.path.join(HEIGHT_DIR, f'level_{level}_surface.png'), dpi=150)
            plt.close()
            
            # Part (c): Analysis
            sub_data = data[
                SUB_REGION["y_start"]:SUB_REGION["y_end"],
                SUB_REGION["x_start"]:SUB_REGION["x_end"]
            ]
            subregion_avg.append(np.mean(sub_data))
            
        except Exception as e:
            print(f"Error processing level {level}: {str(e)}")
            subregion_avg.append(np.nan)

    # Plot analysis results
    plt.figure(figsize=(10, 6))
    plt.plot(range(11), subregion_avg, 'bo-', markersize=8)
    plt.xlabel('Energy Level Index', fontsize=12)
    plt.ylabel('Average LDOS', fontsize=12)
    plt.title('Sub-region LDOS Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(11))
    plt.savefig(os.path.join(ROOT_DIR, 'subregion_analysis.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    process_files()
    print("Processing complete! Check output folders.")
