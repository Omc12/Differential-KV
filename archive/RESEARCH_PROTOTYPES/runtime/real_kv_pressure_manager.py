import torch
import numpy as np

class RealKVPressureManager:
    """
    Generates realistic long-context prompts to pressure the KV cache.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_long_prompt(self, target_tokens: int) -> str:
        """
        Creates a long prompt by repeating a codebase or document.
        """
        base_text = "The Differential KV system implements a sparse retrieval mechanism for the KV cache. "
        base_text += "By using anchors and low-rank deltas, it minimizes VRAM usage. "
        
        # Approximate tokens in base_text
        base_tokens = len(self.tokenizer.encode(base_text))
        repeats = target_tokens // base_tokens
        
        full_text = base_text * repeats
        # Ensure we have a question at the end to trigger generation
        full_text += "\n\nQuestion: Based on the text above, how does Differential KV optimize memory?\nAssistant:"
        
        return full_text

    def get_vram_limit_info(self):
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0}
            
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        
        return {
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - allocated
        }
