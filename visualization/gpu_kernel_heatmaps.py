import matplotlib.pyplot as plt
import numpy as np
import os

def plot_gpu_kernel_heatmap(occupancy_data: np.ndarray, save_path: str):
    """
    Visualizes GPU SM occupancy as a heatmap.
    """
    # Reshape into a 2D grid for visualization
    side = int(np.ceil(np.sqrt(len(occupancy_data))))
    grid = np.zeros((side, side))
    grid.flat[:len(occupancy_data)] = occupancy_data
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Kernel Launch Count")
    plt.title("GPU SM Occupancy & Sparse Kernel Distribution")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    data = np.random.randint(0, 100, 108) # Mock A100 SM count
    plot_gpu_kernel_heatmap(data, "results/phase7_5/gpu_kernel_heatmap.png")
