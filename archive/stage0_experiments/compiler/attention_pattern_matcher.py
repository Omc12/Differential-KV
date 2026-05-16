import torch.nn as nn
from typing import Optional, List, Type

class AttentionPatternMatcher:
    """Matches attention modules across different model architectures."""
    
    PATTERNS = {
        "LlamaAttention": ["LlamaAttention"],
        "MistralAttention": ["MistralAttention"],
        "Qwen2Attention": ["Qwen2Attention"],
        "Phi3Attention": ["Phi3Attention"],
        "GemmaAttention": ["GemmaAttention"]
    }
    
    @staticmethod
    def is_attention_module(module: nn.Module) -> bool:
        class_name = module.__class__.__name__
        for pattern_list in AttentionPatternMatcher.PATTERNS.values():
            if class_name in pattern_list:
                return True
        return False
    
    @staticmethod
    def get_attention_type(module: nn.Module) -> Optional[str]:
        class_name = module.__class__.__name__
        for attn_type, pattern_list in AttentionPatternMatcher.PATTERNS.items():
            if class_name in pattern_list:
                return attn_type
        return None

    @staticmethod
    def find_all_attention(model: nn.Module) -> List[tuple]:
        """Returns list of (name, module) for all attention layers."""
        attention_layers = []
        for name, module in model.named_modules():
            if AttentionPatternMatcher.is_attention_module(module):
                attention_layers.append((name, module))
        return attention_layers
