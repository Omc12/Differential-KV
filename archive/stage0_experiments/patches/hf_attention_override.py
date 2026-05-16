"""
patches/hf_attention_override.py

Infrastructure for monkey-patching HuggingFace attention layers with NCAA.
Supports Llama-3, Qwen2, and Mistral.
"""

import torch
import torch.nn as nn
import importlib
from typing import Optional, Tuple, Dict, Any
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from patches.native_attention_patch import NativeAttentionPatch

def patch_hf_attention(model: nn.Module, config: Dict[str, Any]):
    """
    Traverses the model and replaces standard attention layers with patched versions.
    """
    patched_count = 0
    
    # Identify model type
    model_type = getattr(model.config, "model_type", None)
    print(f"Patching model of type: {model_type}")
    
    for name, module in model.named_modules():
        if isinstance(module, (LlamaAttention, MistralAttention, Qwen2Attention)):
            # Create the patch
            patch = NativeAttentionPatch(
                original_attention=module,
                layer_idx=patched_count,
                config=config
            )
            
            # Replace forward method (monkey-patch)
            # We don't replace the module itself to preserve state/params, 
            # just the forward pass logic.
            module.forward = patch.forward
            patched_count += 1
            
    print(f"Successfully patched {patched_count} attention layers.")
    return model

def unpatch_hf_attention(model: nn.Module):
    """
    Restores original forward methods.
    (Requires original forward to be stored during patching)
    """
    # This would require tracking original forwards. 
    # For Phase 31, we assume persistent patching.
    pass
