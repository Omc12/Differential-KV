import torch

class LocalStructureTracker:
    """
    Tracks local token clusters to detect structural anchors.
    """

    def __init__(self, window_size=5):
        self.window_size = window_size

    def find_clusters(self, importance):
        """
        Identifies regions of high importance (clusters) rather than single tokens.
        """
        # Apply 1D max pooling or convolution to find 'peaks' in importance
        # importance: [batch, seq_len]
        if importance.shape[-1] < self.window_size:
            return importance
            
        # Pad and use moving average to smooth
        padding = self.window_size // 2
        importance_padded = torch.nn.functional.pad(importance.unsqueeze(1), (padding, padding), mode='reflect')
        smoothed = torch.nn.functional.avg_pool1d(importance_padded, kernel_size=self.window_size, stride=1)
        
        return smoothed.squeeze(1)

if __name__ == "__main__":
    tracker = LocalStructureTracker(window_size=3)
    imp = torch.tensor([[0.1, 0.9, 0.8, 0.2, 0.1]])
    clusters = tracker.find_clusters(imp)
    print(f"Original importance: {imp.tolist()}")
    print(f"Smoothed clusters: {clusters.tolist()}")
