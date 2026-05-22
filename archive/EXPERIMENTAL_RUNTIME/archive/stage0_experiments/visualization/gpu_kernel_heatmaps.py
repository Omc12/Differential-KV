import matplotlib.pyplot as plt
import numpy as np

def plot_gpu_kernel_heatmaps(occupancy_data, save_path="reports/gpu_occupancy.png"):
    """
    Plots SM occupancy heatmaps for sparse kernels.
    """
    # Reshape SM data into a grid (e.g., 12x9 for 108 SMs)
    grid_size = int(np.sqrt(len(occupancy_data)))
    if grid_size**2 != len(occupancy_data):
        # Adjust to closest rectangle
        rows = 9
        cols = 12
    else:
        rows = cols = grid_size
        
    heatmap = occupancy_data.reshape((rows, cols))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Occupancy %")
    plt.title("GPU SM Occupancy Heatmap (Sparse Kernel Execution)")
    plt.xlabel("SM Column")
    plt.ylabel("SM Row")
    
    plt.savefig(save_path)
    print(f"Saved occupancy heatmap to {save_path}")

if __name__ == "__main__":
    # Sample data for 108 SMs (A100/H100)
    data = np.random.rand(108)
    plot_gpu_kernel_heatmaps(data)
