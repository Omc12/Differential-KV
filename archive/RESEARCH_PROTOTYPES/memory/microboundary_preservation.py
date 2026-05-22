import torch

class MicroboundaryPreserver:
    """
    PHASE 18.9C: Microboundary Preservation.
    Extremely localized protection for separators and delimiters within symbolic spans.
    Ensures that '-' in 'ALPHA-9' is not pruned.
    """
    def __init__(self, separators=['-', '_', '.', '/', ':', '@', '#', '$', '%', '^', '&', '*', '+', '=']):
        self.separators = separators

    def protect_microboundaries(self, input_ids, tokenizer, seq_len):
        """
        Identifies and masks microboundaries (delimiters).
        """
        mask = torch.zeros(seq_len, device=input_ids.device, dtype=torch.bool)
        # Identify separator tokens
        for sep in self.separators:
            sep_ids = tokenizer.encode(sep, add_special_tokens=False)
            for sid in sep_ids:
                matches = torch.where(input_ids[0] == sid)[0]
                mask[matches] = True
        
        # Also protect transition points where character type changes (e.g. Letter to Digit)
        # This is a bit complex for a micro-preserver, so we'll stick to delimiters for now.
        return mask
