import torch
import torch.nn.functional as F

class PersistentRelevanceTracker:
    """
    PHASE 18.8A: Contextual Persistence Engine.
    Identifies memory-important regions based on semantic recurrence and causal influence.
    FORBIDDEN: Heuristics based on capitalization or special characters.
    """
    def __init__(self, window_size=512, decay=0.95):
        self.window_size = window_size
        self.decay = decay
        self.relevance_map = {} # (chunk_id, token_idx) -> score

    def update_relevance(self, hidden_states, input_ids, chunk_idx):
        """
        Calculates relevance based on semantic centrality and recurrence.
        """
        # hidden_states: [1, seq_len, hidden_dim]
        # input_ids: [1, seq_len]
        
        seq_len = hidden_states.size(1)
        
        # 1. Semantic Centrality
        norm_states = F.normalize(hidden_states, p=2, dim=-1) # [1, seq_len, dim]
        similarity = torch.matmul(norm_states, norm_states.transpose(1, 2)) # [1, seq_len, seq_len]
        centrality = similarity.mean(dim=-1).squeeze(0) # [seq_len]
        
        # 2. Recurrence
        recurrence = torch.zeros(seq_len, device=hidden_states.device)
        if seq_len > 1:
            for i in range(seq_len):
                if i > 0:
                    sim_to_prev = similarity[0, i, :i]
                    if sim_to_prev.numel() > 0:
                        recurrence[i] = sim_to_prev.max()
        
        # Combine signals
        # Normalize to [0, 1]
        centrality = (centrality - centrality.min()) / (centrality.max() - centrality.min() + 1e-6)
        recurrence = (recurrence - recurrence.min()) / (recurrence.max() - recurrence.min() + 1e-6)
        
        relevance_scores = (0.6 * centrality + 0.4 * recurrence)
        
        # Store with decay for persistence across chunks
        for i in range(seq_len):
            key = (chunk_idx, i)
            self.relevance_map[key] = relevance_scores[i].item()
            
        return relevance_scores

    def get_high_relevance_spans(self, chunk_idx, threshold=0.6):
        """
        Returns spans that exhibit persistent relevance.
        Threshold lowered to 0.6 for better recall.
        """
        spans = []
        current_span = None
        
        for i in range(self.window_size):
            score = self.relevance_map.get((chunk_idx, i), 0)
            if score >= threshold:
                if current_span is None:
                    current_span = [i, i]
                else:
                    current_span[1] = i
            else:
                if current_span:
                    spans.append(tuple(current_span))
                    current_span = None
        
        if current_span:
            spans.append(tuple(current_span))
            
        return spans
