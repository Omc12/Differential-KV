import os
"""
runtime/hf_diffkv_wrapper.py

HuggingFace model wrapper for Differential KV.
Integrates KVRuntimeManager with AutoModelForCausalLM.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.kv_runtime_manager import KVRuntimeManager, KVBlock
from compression.lowrank import compress_lowrank
from compression.sparse_repair import compress_lowrank_sparse
from compression.quantization import quantize_int8
from runtime.triton_diffkv import TritonDiffKV

class DiffKVHFWrapper:
    """
    Wraps a HuggingFace model to use Differential KV cache.
    """
    def __init__(
        self, 
        model_id: str,
        config: Dict[str, Any],
        device: str = "cuda",
        quantization_config: Any = None
    ):
        self.model_id = model_id
        self.config = config
        self.device = device
        self.mode = config.get("mode", "fp16")
        self.block_size = config.get("block_size", 64)
        self.rank = config.get("rank", 16)
        
        print(f"Loading model {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=device,
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.heads
        
        self.manager = KVRuntimeManager(self.num_layers, self.heads, self.head_dim, device=device)
        self.active_session = None
        
        # Temporary storage for tokens in the current block (before compression)
        self.current_blocks: Dict[int, List[torch.Tensor]] = {i: [] for i in range(self.num_layers)}
        self.current_anchors: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs.input_ids
        generated = input_ids[0].tolist()
        
        outputs = self.model(input_ids=input_ids, use_cache=True)
        self._update_manager(outputs.past_key_values)
        logits = outputs.logits[:, -1, :]
        
        for i in range(max_new_tokens):
            next_token_id = torch.argmax(logits, dim=-1)
            generated.append(next_token_id.item())
            input_ids = next_token_id.unsqueeze(0)
            
            outputs = self.model(input_ids=input_ids, use_cache=True)
            logits = outputs.logits[:, -1, :]
            self._update_manager(outputs.past_key_values)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)

    def switch_session(self, session_id: str):
        self.manager.switch_session(session_id)
        self.active_session = session_id

    def forward_step(self, input_ids: torch.Tensor, session_id: Optional[str] = None) -> torch.Tensor:
        if session_id:
            self.switch_session(session_id)
            
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
            self._update_manager(outputs.past_key_values, session_id)
            return outputs.logits[:, -1, :]

    def _update_manager(self, past_key_values, session_id: Optional[str] = None):
        if past_key_values is None:
            return
            
        for layer_idx, (k, v) in enumerate(past_key_values):
            # For EOM validation, we simplified the manager to focus on serving overhead.
            # In a real system, this would compress and add blocks.
            self.manager.update_layer(layer_idx, k, v, session_id)

    def _custom_sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
