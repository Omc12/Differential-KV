import torch

class SinkTokenTracker:
    """
    Monitors attention distributions to identify stable attention sinks
    and track their importance over time.
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.importance_history = []

    def track_importance(self, attention_weights):
        """
        attention_weights: [batch, heads, query_len, key_len]
        Calculates the mean attention received by each token.
        """
        # Sum across heads and queries to get importance per token in context
        # shape: [batch, key_len]
        importance = attention_weights.mean(dim=(1, 2))
        
        self.importance_history.append(importance.detach().cpu())
        if len(self.importance_history) > self.window_size:
            self.importance_history.pop(0)
            
        return importance

    def get_stable_sinks(self, threshold=0.1):
        """
        Identifies tokens that consistently receive high attention.
        """
        if not self.importance_history:
            return []
            
        mean_importance = torch.stack(self.importance_history).mean(dim=0)
        # Find indices where mean importance is above threshold
        sink_indices = torch.where(mean_importance > threshold)[1].tolist()
        
        return list(set(sink_indices))

if __name__ == "__main__":
    tracker = SinkTokenTracker()
    # Simulated attention: high attention on token 0
    attn = torch.zeros(1, 8, 1, 10)
    attn[:, :, :, 0] = 0.5
    attn[:, :, :, 1:] = 0.5 / 9
    
    tracker.track_importance(attn)
    sinks = tracker.get_stable_sinks(threshold=0.2)
    print(f"Detected stable sinks: {sinks}")
