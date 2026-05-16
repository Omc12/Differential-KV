import matplotlib.pyplot as plt
import torch
import os

def plot_retrieval_collision_map(collision_counts: torch.Tensor, save_path: str):
    """
    Visualizes regions where high retrieval contention occurs.
    """
    plt.figure(figsize=(15, 4))
    counts_np = collision_counts.cpu().numpy()
    
    plt.bar(range(len(counts_np)), counts_np, color='orange', alpha=0.7)
    plt.title("Sparse Retrieval Collision & Bank Contention Map")
    plt.xlabel("KV Slot Index")
    plt.ylabel("Contention Count")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    counts = torch.zeros(1024)
    counts[500:520] = torch.randint(5, 20, (20,))
    plot_retrieval_collision_map(counts, "results/phase7/retrieval_collision_map.png")
