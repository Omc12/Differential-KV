"""
runtime/hf_dkv_wrapper.py

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
from runtime.triton_dkv import TritonDKV

class DKVHFWrapper:
    """
    Wraps a HuggingFace model to use Differential KV cache.
    """
    def __init__(
        self, 
        model_id: str,
        config: Dict[str, Any],
        device: str = "cuda"
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
            trust_remote_code=True
        )
        self.model.eval()
        
        self.manager = KVRuntimeManager(config, device=device)
        self.num_layers = self.model.config.num_hidden_layers
        
        # Temporary storage for tokens in the current block (before compression)
        self.current_blocks: Dict[int, List[torch.Tensor]] = {i: [] for i in range(self.num_layers)}
        self.current_anchors: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # We'll use a simple loop instead of model.generate to have full control over the KV cache
        generated = input_ids[0].tolist()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # In a real integrated system, we'd reconstruct KV here
            # For this wrapper, we simulate the overhead and VRAM residency
            
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated.append(next_token_id.item())
            
            # Update our custom KV manager (simulated)
            self._update_manager(outputs.past_key_values)
            
            # For the next step, we use the standard past_key_values for now,
            # but in Task 3/4 we will measure the reconstruction impact.
            past_key_values = outputs.past_key_values
            input_ids = next_token_id.unsqueeze(0)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
        return self.tokenizer.decode(generated)

    def _update_manager(self, past_key_values):
        """
        Extract the latest KV from HF output and put it into our manager.
        """
        if past_key_values is None:
            return
            
        for layer_idx, (k, v) in enumerate(past_key_values):
            # k, v: [batch, heads, seq_len, head_dim]
            # Convert to [seq, 2, heads, dim]
            # stack k, v -> [batch, 2, heads, seq, dim]
            kv_all = torch.stack([k, v], dim=1).squeeze(0) # [2, heads, seq, dim]
            kv_all = kv_all.permute(2, 0, 1, 3) # [seq, 2, heads, dim]
            
            # Add each token to current block
            for i in range(kv_all.shape[0]):
                kv_token = kv_all[i:i+1] # [1, 2, heads, dim]
                self.current_blocks[layer_idx].append(kv_token)
                
                # If block size reached, compress
                if len(self.current_blocks[layer_idx]) >= self.block_size:
                    self._compress_current_block(layer_idx)

    def _compress_current_block(self, layer_idx: int):
        tokens = self.current_blocks[layer_idx]
        anchor = tokens[0] # [1, 2, heads, head_dim]
        deltas = torch.cat(tokens[1:], dim=0) # [block_size-1, 2, heads, head_dim]
        
        # Prepare for low-rank
        n, _, h, d = deltas.shape
        deltas_flat = deltas.view(n, -1) # [n, 2*h*d]
        
        if self.mode == "lowrank" or self.mode == "lowrank_sparse":
            if self.mode == "lowrank_sparse":
                lrs = compress_lowrank_sparse(deltas_flat.float(), self.rank, self.config.get("sparse_ratio", 0.01))
                lr = lrs.low_rank
                s_idx, s_val = lrs.sparse_indices, lrs.sparse_values
            else:
                lr = compress_lowrank(deltas_flat.float(), self.rank)
                s_idx, s_val = None, None
            
            block = KVBlock(
                anchor_idx=0, 
                anchor_kv=anchor.squeeze(0),
                U=lr.U,
                V=lr.V,
                scale=lr.scale,
                sparse_indices=s_idx,
                sparse_values=s_val,
                token_indices=list(range(len(tokens))),
                mode=self.mode
            )
        elif self.mode == "int8":
            q_deltas = quantize_int8(deltas_flat.float())
            block = KVBlock(
                anchor_idx=0,
                anchor_kv=anchor.squeeze(0),
                q_deltas=q_deltas,
                token_indices=list(range(len(tokens))),
                mode="int8"
            )
        elif self.mode == "shared_basis":
            # Determine if we need a new basis
            # For simplicity, we'll use a Layer-Shared basis: 
            # The first block of the layer defines the basis for all subsequent blocks.
            basis_id = 0 # Default basis ID
            if basis_id not in self.manager.basis_cache.get(layer_idx, {}):
                # Extract basis from this block
                V = extract_basis(deltas_flat.float(), self.rank)
                self.manager.add_basis(layer_idx, basis_id, V)
            else:
                V = self.manager.basis_cache[layer_idx][basis_id]
            
            # Compress using shared basis
            sb = compress_shared_basis(deltas_flat.float(), V, basis_id, sparse_ratio=self.config.get("sparse_ratio", 0.0))
            
            block = KVBlock(
                anchor_idx=0,
                anchor_kv=anchor.squeeze(0),
                U=sb.U,
                basis_id=sb.basis_id,
                scale=sb.scale,
                sparse_indices=sb.sparse_indices,
                sparse_values=sb.sparse_values,
                token_indices=list(range(len(tokens))),
                mode="shared_basis"
            )
        elif self.mode == "fp16":
            # Store everything as raw deltas (simulated)
            block = KVBlock(
                anchor_idx=0,
                anchor_kv=anchor.squeeze(0),
                q_deltas=deltas, # Just store the deltas as-is for residency measurement
                token_indices=list(range(len(tokens))),
                mode="fp16"
            )
        else:
            # Default to periodic/raw
            block = KVBlock(
                anchor_idx=0,
                anchor_kv=anchor.squeeze(0),
                token_indices=list(range(len(tokens))),
                mode="periodic"
            )
            
        self.manager.add_block(layer_idx, block)
            
        self.current_blocks[layer_idx] = []
