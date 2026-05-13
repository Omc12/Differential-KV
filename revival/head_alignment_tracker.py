import torch

class HeadAlignmentTracker:
    """
    Tracks how well different attention heads are aligned in their token selection.
    """

    def __init__(self):
        self.alignment_scores = []

    def calculate_alignment(self, pruning_masks):
        """
        pruning_masks: [num_heads, seq_len] (bool)
        Calculates Jaccard similarity or overlap between heads.
        """
        num_heads = pruning_masks.shape[0]
        if num_heads < 2:
            return 1.0
            
        total_overlap = 0.0
        count = 0
        
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                intersection = (pruning_masks[i] & pruning_masks[j]).sum().float()
                union = (pruning_masks[i] | pruning_masks[j]).sum().float()
                overlap = intersection / (union + 1e-9)
                total_overlap += overlap.item()
                count += 1
                
        avg_alignment = total_overlap / count
        self.alignment_scores.append(avg_alignment)
        return avg_alignment

if __name__ == "__main__":
    tracker = HeadAlignmentTracker()
    masks = torch.tensor([
        [True, True, False, False],
        [True, False, True, False],
        [True, True, True, False]
    ])
    
    alignment = tracker.calculate_alignment(masks)
    print(f"Mean Head Alignment: {alignment:.4f}")
