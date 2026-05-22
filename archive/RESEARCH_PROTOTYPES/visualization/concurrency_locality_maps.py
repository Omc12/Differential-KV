import matplotlib.pyplot as plt
import numpy as np

def plot_concurrency_locality_maps(locality_matrix, user_labels, save_path="reports/concurrency_locality.png"):
    """
    Visualizes memory locality and interference between concurrent users.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(locality_matrix, cmap="coolwarm")
    plt.colorbar(label="Interference Index")
    
    plt.xticks(range(len(user_labels)), user_labels)
    plt.yticks(range(len(user_labels)), user_labels)
    
    plt.title("Multi-User Retrieval Locality & Interference Map")
    plt.xlabel("User ID")
    plt.ylabel("User ID")
    
    plt.savefig(save_path)
    print(f"Saved locality map to {save_path}")

if __name__ == "__main__":
    users = [f"User {i}" for i in range(8)]
    # Diagonal should be low interference (1.0), off-diagonal should be interference level
    data = np.eye(8) + 0.1 * np.random.rand(8, 8)
    plot_concurrency_locality_maps(data, users)
