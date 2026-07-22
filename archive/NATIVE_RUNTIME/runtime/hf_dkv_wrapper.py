import os
"""
runtime/hf_dkv_wrapper.py

HuggingFace model wrapper for Differential KV.
Integrates KVRuntimeManager with AutoModelForCausalLM.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.kv_runtime_manager import KVRuntimeManager

from runtime.triton_dkv import TritonDKV
from runtime.dkv_attention import apply_dkv_attention_patch

class DKVHFWrapper:
    """
    Wraps a HuggingFace model to use Differential KV cache.
    """
    def __init__(
        self, 
        model_id: str,
        config: Dict[str, Any],
        device: str = "cuda",
        quantization_config: Any = None,
        torch_dtype: torch.dtype = torch.float16
    ):
        self.model_id = model_id
        self.config = config
        self.device = device
        self.mode = config.get("mode", "fp16")
        self.block_size = config.get("block_size", 64)
        self.rank = config.get("rank", 16)
        
        print(f"Loading model {model_id} (dtype={torch_dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
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
        
        # Apply Differential KV Attention Interception!
        apply_dkv_attention_patch(self.model, self.manager)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs.input_ids
        generated = input_ids[0].tolist()

        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [1, vocab]

        for _ in range(max_new_tokens):
            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok_id in set(generated[-64:]):
                    if tok_id < logits.shape[-1]:
                        if logits[0, tok_id] > 0:
                            logits[0, tok_id] /= repetition_penalty
                        else:
                            logits[0, tok_id] *= repetition_penalty

            # Sample
            if temperature <= 0.01:
                next_id = torch.argmax(logits, dim=-1)
            else:
                scaled = logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                if top_p < 1.0:
                    s_probs, s_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(s_probs, dim=-1)
                    mask = (cum - s_probs) > top_p
                    s_probs[mask] = 0.0
                    s_probs = s_probs / s_probs.sum(dim=-1, keepdim=True)
                    sample = torch.multinomial(s_probs, 1)
                    next_id = s_idx.gather(-1, sample).squeeze(-1)
                else:
                    next_id = torch.multinomial(probs, 1).squeeze(-1)

            generated.append(next_id.item())
            if next_id.item() == self.tokenizer.eos_token_id:
                break

            input_ids = next_id.unsqueeze(0)
            outputs = self.model(
                input_ids=input_ids, past_key_values=past_kv, use_cache=True
            )
            logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            self._update_manager(past_kv)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def _custom_sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
