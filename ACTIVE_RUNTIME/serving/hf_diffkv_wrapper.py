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
from native_core.kv_runtime_manager import KVRuntimeManager

from native_core.sparse_decode.triton_diffkv import TritonDiffKV
from runtime.diffkv_attention import apply_diffkv_attention_patch

class DiffKVHFWrapper:
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
        self.micro_block_size = config.get("micro_block_size", 16)
        
        print(f"Loading model {model_id} (dtype={torch_dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._alphanumeric_tokens = {}
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
        
        self.kv_heads = getattr(self.model.config, "num_key_value_heads", self.heads)
        self.serving_mode = config.get("serving_mode", "balanced")
        self.manager = KVRuntimeManager(
            self.num_layers,
            self.kv_heads,
            self.head_dim,
            device=device,
            rank=self.rank,
            micro_block_size=self.micro_block_size,
            serving_mode=self.serving_mode
        )
        self.active_session = None
        
        # Apply Differential KV Attention Interception!
        apply_diffkv_attention_patch(self.model, self.manager)

        # Optional Torch Compile JIT wrapper for auto-fusion
        if os.environ.get("DIFFKV_USE_TORCH_COMPILE", "0") == "1":
            print("[DiffKV] Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def stop(self):
        """Cleanly release all resources and stop background worker threads."""
        if hasattr(self, "manager") and self.manager is not None:
            if hasattr(self.manager, "clear"):
                self.manager.clear()
            if hasattr(self.manager, "pager") and self.manager.pager is not None:
                if hasattr(self.manager.pager, "stop"):
                    self.manager.pager.stop()
            if hasattr(self.manager, "_compressor") and self.manager._compressor is not None:
                self.manager._compressor.stop()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
    ):
        # Clear previous session cache to prevent memory/block leaks
        session_id = self.active_session or "default"
        self.manager.clear_session(session_id)

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
                        # Skip punctuation/newlines/whitespace to avoid suppressing lists/bullets/formatting
                        is_alnum = self._alphanumeric_tokens.get(tok_id)
                        if is_alnum is None:
                            tok_text = self.tokenizer.decode([tok_id], skip_special_tokens=True)
                            is_alnum = any(c.isalnum() for c in tok_text)
                            self._alphanumeric_tokens[tok_id] = is_alnum
                        
                        if not is_alnum:
                            continue

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

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def switch_session(self, session_id: str):
        self.active_session = session_id

    def _custom_sample(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
