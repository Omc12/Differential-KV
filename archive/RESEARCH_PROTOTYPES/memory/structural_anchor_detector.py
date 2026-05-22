import torch

class StructuralAnchorDetector:
    """
    PHASE 18.9A: Structural Anchor Detection.
    Optimized version using vectorized matching for stable structural markers.
    Detects boundaries that naturally organize memory flow.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # MANDATORY STRUCTURAL ANCHORS:
        # newline boundaries, "Context:" markers, section headers, 
        # instruction transitions, role boundaries, conversational pivots, retrieval delimiters.
        self.anchor_texts = [
            '\n', '\n\n', 
            'Context:', '### Context', '## Context',
            'Question:', 'Answer:', 
            'Instruction:', 'Task:',
            '<|im_start|>', '<|im_end|>',
            'User:', 'Assistant:', 'System:',
            '---', '***', '===',
            '[', ']', '{', '}', '(', ')', ':', ';', '|'
        ]
        self.anchor_ids = []
        self._init_token_ids()

    def _init_token_ids(self):
        for text in self.anchor_texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                self.anchor_ids.append(torch.tensor(ids))

    def detect_anchors(self, input_ids):
        """
        Returns a mask of anchor positions.
        """
        seq_len = input_ids.size(1)
        device = input_ids.device
        mask = torch.zeros(seq_len, device=device, dtype=torch.bool)
        
        for ids in self.anchor_ids:
            ids = ids.to(device)
            n = len(ids)
            if n == 1:
                mask |= (input_ids[0] == ids[0])
            elif n <= seq_len:
                # Vectorized sequence matching for multi-token anchors
                # We use unfold to get all windows of size n
                windows = input_ids[0].unfold(0, n, 1)
                matches = (windows == ids).all(dim=1)
                # Mark all positions within the match
                for i in range(n):
                    mask[i:seq_len-n+i+1] |= matches
                        
        return mask

    def get_anchor_indices(self, input_ids):
        mask = self.detect_anchors(input_ids)
        return torch.where(mask)[0]
