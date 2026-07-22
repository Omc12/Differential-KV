import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

class LiveKVInterceptor(nn.Module):
    """
    Intercepts KV tensors during the live forward pass of a transformer.
    Allows DKV to observe and modify the KV stream without breaking the model.
    """
    def __init__(self, original_module: nn.Module, layer_idx: int, runtime_manager: Any):
        super().__init__()
        self.original_module = original_module
        self.layer_idx = layer_idx
        self.manager = runtime_manager
        
    def forward(self, *args, **kwargs) -> Any:
        # Execute original forward pass
        output = self.original_module(*args, **kwargs)
        
        # Intercept KV if present in output
        # Standard HF models return (logits, past_key_values, ...) or just (hidden_states, past_key_values)
        if isinstance(output, tuple) and len(output) > 1:
            pk = output[1]
            if pk is not None:
                # Log or process the live KV
                self.manager.log_kv_state(self.layer_idx, pk)
                
        return output

    @classmethod
    def wrap_model(cls, model: nn.Module, manager: Any):
        """
        Walks through the model and wraps attention modules with the interceptor.
        """
        for name, module in model.named_modules():
            if "self_attn" in name or "Attention" in name:
                # Find the parent module to replace the child
                pass
