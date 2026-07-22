"""
evaluation/generation_drift.py — Phase 2.5 Objective 1

Measures generation drift: how much does greedy decoding diverge
when using DKV-reconstructed KV caches vs full FP16 baseline?

Metrics:
  - First divergence token (longer = better)
  - Token sequence overlap (BLEU-like n-gram)
  - Semantic similarity (cosine of average embeddings, optional)
  - Edit distance (Levenshtein ratio on output tokens)
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GenerationResult:
    label: str
    text_type: str
    prompt_preview: str
    # Divergence
    first_divergence_token: int     # -1 if identical
    sequence_length: int
    token_overlap: float            # fraction of shared tokens (set overlap)
    edit_distance_ratio: float      # 0=identical, 1=completely different
    # Sequences
    baseline_tokens: List[int]
    dkv_tokens: List[int]
    # Memory
    compression_ratio: float

    def is_identical(self) -> bool:
        return self.first_divergence_token == -1

    def to_dict(self) -> Dict:
        return {
            "label":                   self.label,
            "text_type":               self.text_type,
            "first_divergence_token":  self.first_divergence_token,
            "sequence_length":         self.sequence_length,
            "token_overlap":           round(self.token_overlap, 4),
            "edit_distance_ratio":     round(self.edit_distance_ratio, 4),
            "identical":               self.is_identical(),
            "compression_ratio":       round(self.compression_ratio, 4),
        }


def _levenshtein_ratio(a: List[int], b: List[int]) -> float:
    """Levenshtein distance / max(len(a), len(b))."""
    if not a and not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev[j-1] + cost)
    return dp[m] / max(n, m)


def _token_overlap(a: List[int], b: List[int]) -> float:
    """Fraction of shared tokens (set-based)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


class GenerationDriftEvaluator:
    """
    Generates text with both baseline and DKV-reconstructed KV caches
    and measures divergence.

    Parameters
    ----------
    model_name   : str
    device       : str
    max_new_tokens : int — how many tokens to generate for comparison
    max_context_len : int — prefix length (KV to compress)
    """

    def __init__(self, model_name: str = "gpt2", device: str = "auto",
                 max_new_tokens: int = 50, max_context_len: int = 128):
        self.model_name      = model_name
        self.device          = device
        self.max_new_tokens  = max_new_tokens
        self.max_context_len = max_context_len
        self._model          = None
        self._tokenizer      = None

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_map = {
            "gpt2":      "gpt2",
            "gpt2-med":  "gpt2-medium",
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "opt-125m":  "facebook/opt-125m",
        }
        mid = model_map.get(self.model_name, self.model_name)
        print(f"  [GenerationDrift] Loading {mid}")
        self._tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16,
            device_map=self.device, trust_remote_code=True
        )
        self._model.eval()

    def _greedy_generate(self, prefix_ids, past_kv, max_new: int) -> List[int]:
        """Greedy decode `max_new` tokens starting from compressed past_kv."""
        generated = []
        current_past = past_kv
        current_ids  = prefix_ids[:, -1:]  # last prefix token as seed

        with torch.no_grad():
            for _ in range(max_new):
                out = self._model(input_ids=current_ids,
                                  past_key_values=current_past, use_cache=True)
                next_tok = out.logits[:, -1, :].argmax(-1)  # [1]
                generated.append(next_tok.item())
                current_ids  = next_tok.unsqueeze(0)
                current_past = out.past_key_values
                if next_tok.item() == self._tokenizer.eos_token_id:
                    break

        return generated

    def _compress_kv(self, past_kv, strategy_label: str):
        """Compress past_kv with DKV; returns reconstructed tuple + stats."""
        from anchor_logic.anchor_manager import AnchorManager
        from anchor_logic.strategies import PeriodicAnchorStrategy
        from anchor_logic.adaptive_policies import EMAPolicy, RollingVariancePolicy
        from reconstruction.reconstructor import KVReconstructor

        strategy_map = {
            "periodic_64":  lambda: PeriodicAnchorStrategy(interval=64),
            "periodic_128": lambda: PeriodicAnchorStrategy(interval=128),
            "ema_balanced": lambda: EMAPolicy(alpha=0.1, sensitivity_factor=2.5,
                                              max_interval=256, min_interval=8),
            "rolling_k3":   lambda: RollingVariancePolicy(k=3.0, window_size=64,
                                                          max_interval=256, min_interval=8),
            "lowrank_1":    lambda: "lowrank_1",
            "lowrank_2":    lambda: "lowrank_2",
            "lowrank_4":    lambda: "lowrank_4",
            "lowrank_8":    lambda: "lowrank_8",
            "lowrank_16":   lambda: "lowrank_16",
            "lowrank_32":   lambda: "lowrank_32",
        }

        reconstructed = []
        total_orig = total_comp = 0

        for k, v in past_kv:
            if k.shape[1] < k.shape[2]:
                k_n = k.squeeze(0).permute(1, 0, 2).cpu().float()
                v_n = v.squeeze(0).permute(1, 0, 2).cpu().float()
            else:
                k_n = k.squeeze(0).cpu().float()
                v_n = v.squeeze(0).cpu().float()

            kv_seq = torch.stack([k_n, v_n], dim=1)
            seq_len = kv_seq.shape[0]

            strat   = strategy_map.get(strategy_label, strategy_map["periodic_64"])()
            
            total_orig += seq_len * kv_seq.shape[1] * kv_seq.shape[2] * kv_seq.shape[3] * 2
            
            if isinstance(strat, str) and strat.startswith("lowrank_"):
                from compression.lowrank import compress_kv_sequence_lowrank, decompress_kv_sequence_lowrank
                rank = int(strat.split("_")[1])
                anchor_interval = 64
                anchor_positions = list(range(0, seq_len, anchor_interval))
                blocks, kv_anchors = compress_kv_sequence_lowrank(kv_seq, anchor_positions, rank)
                kv_r = decompress_kv_sequence_lowrank(blocks, kv_anchors, kv_seq.shape)
                
                from compression.lowrank import estimate_memory
                heads, dim = kv_seq.shape[2], kv_seq.shape[3]
                mem = estimate_memory(seq_len, heads, dim, rank, interval=anchor_interval)
                total_comp += mem["lowrank_bytes"]
            else:
                manager = AnchorManager(strategy=strat)
                stats   = manager.compress(kv_seq)
                recon   = KVReconstructor(manager)
                result  = recon.reconstruct_range(0, seq_len - 1)
                kv_r    = result.kv
                total_comp += stats.total_compressed_bytes

            k_r = kv_r[:, 0, :, :].to(k.device, k.dtype)
            v_r = kv_r[:, 1, :, :].to(v.device, v.dtype)
            if k.shape[1] < k.shape[2]:
                k_r = k_r.permute(1, 0, 2).unsqueeze(0)
                v_r = v_r.permute(1, 0, 2).unsqueeze(0)
            else:
                k_r = k_r.unsqueeze(0)
                v_r = v_r.unsqueeze(0)
            reconstructed.append((k_r, v_r))

        ratio = total_orig / (total_comp + 1e-9)
        return tuple(reconstructed), ratio

    def evaluate(
        self,
        prompts: List[str],
        text_types: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
    ) -> List[GenerationResult]:

        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        strategies = strategies or ["periodic_64", "ema_balanced", "rolling_k3"]
        text_types = text_types or ["unknown"] * len(prompts)
        results    = []

        for prompt, ttype in zip(prompts, text_types):
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=self.max_context_len)
            prefix_ids = inputs["input_ids"].to(self._model.device)
            if prefix_ids.shape[1] < 4:
                continue

            # Baseline generation
            with torch.no_grad():
                base_out = self._model(input_ids=prefix_ids, use_cache=True)
            base_kv     = base_out.past_key_values
            base_tokens = self._greedy_generate(prefix_ids, base_kv, self.max_new_tokens)

            for strat in strategies:
                try:
                    recon_kv, ratio = self._compress_kv(
                        self._get_fresh_kv(prefix_ids), strat
                    )
                    dkv_tokens = self._greedy_generate(prefix_ids, recon_kv,
                                                          self.max_new_tokens)

                    # First divergence
                    first_div = -1
                    for k, (bt, dt) in enumerate(zip(base_tokens, dkv_tokens)):
                        if bt != dt:
                            first_div = k
                            break

                    overlap = _token_overlap(base_tokens, dkv_tokens)
                    edit_r  = _levenshtein_ratio(base_tokens, dkv_tokens)

                    r = GenerationResult(
                        label=strat, text_type=ttype,
                        prompt_preview=prompt[:60],
                        first_divergence_token=first_div,
                        sequence_length=len(base_tokens),
                        token_overlap=overlap,
                        edit_distance_ratio=edit_r,
                        baseline_tokens=base_tokens,
                        dkv_tokens=dkv_tokens,
                        compression_ratio=ratio,
                    )
                    results.append(r)

                    div_str = f"tok {first_div}" if first_div >= 0 else "IDENTICAL"
                    print(f"    [{strat:<16}] div={div_str:<8}  "
                          f"overlap={overlap:.3f}  edit={edit_r:.3f}  ratio={ratio:.2f}x")
                except Exception as e:
                    print(f"    [{strat}] ERROR: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _get_fresh_kv(self, prefix_ids):
        with torch.no_grad():
            out = self._model(input_ids=prefix_ids, use_cache=True)
        return out.past_key_values
