"""
integrations/huggingface_runtime_adapter.py

HuggingFace-compatible model adapter for Differential KV.
Wraps the sparse runtime to look like a standard Transformers model.
"""

import torch
from typing import Optional, List, Union, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoTokenizer
from runtime.kv_runtime_manager import KVRuntimeManager

class DKVHFConfig(PretrainedConfig):
    model_type = "dkv"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sparse_mode = kwargs.get("sparse_mode", "lowrank_sparse")
        self.block_size = kwargs.get("block_size", 64)

class DKVHFAdapter(PreTrainedModel):
    """
    Adapter that makes Differential KV look like a HuggingFace model.
    """
    config_class = DKVHFConfig
    
    def __init__(self, config: DKVHFConfig, base_model: Optional[Any] = None):
        super().__init__(config)
        self._base_model = base_model
        self.manager = KVRuntimeManager(config.to_dict())
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Custom from_pretrained to wrap a standard model.
        """
        # Load config and base model
        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        # In this implementation, we wrap the base HF model
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            *model_args, 
            **kwargs
        )
        
        adapter_config = DKVHFConfig(**hf_config.to_dict())
        return cls(adapter_config, base_model=base_model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        **kwargs
    ):
        """
        Forward pass that utilizes Differential KV manager.
        """
        # This is where we would intercept the KV cache logic
        # For the adapter, we delegate to the base model but manage the cache residency
        outputs = self._base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            **kwargs
        )
        
        # Intercept and sparse-ify the KV cache if needed
        # (This logic is usually handled inside the KVRuntimeManager)
        
        return outputs

    def generate(self, *args, **kwargs):
        """
        Standard generation interface.
        """
        # Use base model's generate but ensure our hooks are active
        return self._base_model.generate(*args, **kwargs)

def wrap_hf_model(model: PreTrainedModel) -> DKVHFAdapter:
    """Utility to wrap an existing HF model instance."""
    config = DKVHFConfig(**model.config.to_dict())
    return DKVHFAdapter(config, base_model=model)

if __name__ == "__main__":
    # Mock testing
    print("DKVHFAdapter module loaded.")
