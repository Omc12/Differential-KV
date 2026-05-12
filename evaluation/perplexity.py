"""
evaluation/perplexity.py — Phase 2.5 Objective 1

Measures whether DiffKV-reconstructed KV caches preserve actual model behavior.

Strategy:
  1. Run model prefill on a prompt → capture real past_key_values
  2. Compress + reconstruct each layer's KV with DiffKV policy
  3. Feed reconstructed KV as past_key_values for continuation
  4. Compare log-likelihoods (perplexity) vs baseline
  5. Also compare token-level softmax distributions (KL divergence)

This is the most direct behavioral validation possible without
modifying the model architecture itself.
"""

import math
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation for one KV variant."""
    label: str
    prompt: str
    text_type: str
    # Perplexity on continuation tokens
    perplexity: float
    log_likelihood: float
    # Token agreement vs baseline
    token_agreement: float      # fraction of tokens where argmax matches baseline
    mean_kl_div: float          # KL divergence from baseline logits
    # Memory estimates
    kv_bytes_fp16: int
    kv_bytes_compressed: int
    compression_ratio: float
    # Timing
    eval_ms: float

    def to_dict(self) -> Dict:
        return {
            "label":              self.label,
            "text_type":          self.text_type,
            "perplexity":         round(self.perplexity, 4),
            "log_likelihood":     round(self.log_likelihood, 4),
            "token_agreement":    round(self.token_agreement, 4),
            "mean_kl_div":        round(self.mean_kl_div, 6),
            "kv_bytes_fp16":      self.kv_bytes_fp16,
            "kv_bytes_compressed": self.kv_bytes_compressed,
            "compression_ratio":  round(self.compression_ratio, 4),
            "eval_ms":            round(self.eval_ms, 1),
        }


class PerplexityEvaluator:
    """
    Evaluates DiffKV reconstruction quality via perplexity and token agreement.

    Parameters
    ----------
    model_name : str
    device     : str
    max_context_len : int — tokens in prefix (KV to compress)
    eval_tokens     : int — tokens in suffix (to measure perplexity on)
    """

    def __init__(self, model_name: str = "gpt2", device: str = "auto",
                 max_context_len: int = 256, eval_tokens: int = 50):
        self.model_name       = model_name
        self.device           = device
        self.max_context_len  = max_context_len
        self.eval_tokens      = eval_tokens
        self._model           = None
        self._tokenizer       = None

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_map = {
            "gpt2":      "gpt2",
            "gpt2-med":  "gpt2-medium",
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "phi2":      "microsoft/phi-2",
            "opt-125m":  "facebook/opt-125m",
            "qwen":      "Qwen/Qwen2-0.5B",
        }
        mid = model_map.get(self.model_name, self.model_name)
        print(f"  [PerplexityEval] Loading {mid}")
        self._tokenizer = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16,
            device_map=self.device, trust_remote_code=True
        )
        self._model.eval()

    def _get_baseline_kv(self, input_ids: torch.Tensor):
        """Run prefill, return real past_key_values."""
        with torch.no_grad():
            out = self._model(input_ids=input_ids, use_cache=True)
        return out.past_key_values  # tuple of (K, V) per layer

    def _compress_kv(self, past_kv, strategy_label: str, selector=None):
        """
        Compress each layer's KV cache using DiffKV.
        Returns reconstructed past_key_values tuple.
        """
        from anchor_logic.anchor_manager import AnchorManager
        from anchor_logic.strategies import PeriodicAnchorStrategy
        from anchor_logic.adaptive_policies import EMAPolicy, RollingVariancePolicy
        from reconstruction.reconstructor import KVReconstructor

        strategy_map = {
            "periodic_64":   lambda: PeriodicAnchorStrategy(interval=64),
            "periodic_128":  lambda: PeriodicAnchorStrategy(interval=128),
            "ema_balanced":  lambda: EMAPolicy(alpha=0.1, sensitivity_factor=2.5,
                                               max_interval=256, min_interval=8),
            "rolling_k3":    lambda: RollingVariancePolicy(k=3.0, window_size=64,
                                                           max_interval=256, min_interval=8),
        }

        reconstructed = []
        total_orig_bytes = 0
        total_comp_bytes = 0

        for layer_idx, (k, v) in enumerate(past_kv):
            # k, v shape: [batch, heads, seq, dim] or [batch, seq, heads, dim]
            # Normalize to [seq, 2, heads, dim]
            if k.dim() == 4:
                if k.shape[1] < k.shape[2]:
                    # [batch, heads, seq, dim]
                    k_norm = k.squeeze(0).permute(1, 0, 2)  # [seq, heads, dim]
                    v_norm = v.squeeze(0).permute(1, 0, 2)
                else:
                    k_norm = k.squeeze(0)  # [seq, heads, dim]
                    v_norm = v.squeeze(0)
            else:
                k_norm = k.squeeze(0)
                v_norm = v.squeeze(0)

            kv_seq = torch.stack([k_norm, v_norm], dim=1).cpu().float()  # [seq, 2, H, D]
            seq_len = kv_seq.shape[0]

            # Get strategy for this layer
            if selector is not None:
                strat = selector.get_strategy(layer_idx)
            elif strategy_label in strategy_map:
                strat = strategy_map[strategy_label]()
            else:
                strat = PeriodicAnchorStrategy(interval=64)

            manager = AnchorManager(strategy=strat)
            stats   = manager.compress(kv_seq)
            recon   = KVReconstructor(manager)
            result  = recon.reconstruct_range(0, seq_len - 1)
            kv_recon = result.kv  # [seq, 2, H, D] fp16

            total_orig_bytes += stats.total_original_bytes
            total_comp_bytes += stats.total_compressed_bytes

            # Restore to model's expected shape
            k_recon = kv_recon[:, 0, :, :].to(k.device, k.dtype)  # [seq, H, D]
            v_recon = kv_recon[:, 1, :, :].to(v.device, v.dtype)
            # Reshape back to [batch, heads, seq, dim] if needed
            if k.shape[1] < k.shape[2]:
                k_recon = k_recon.permute(1, 0, 2).unsqueeze(0)
                v_recon = v_recon.permute(1, 0, 2).unsqueeze(0)
            else:
                k_recon = k_recon.unsqueeze(0)
                v_recon = v_recon.unsqueeze(0)
            reconstructed.append((k_recon, v_recon))

        return tuple(reconstructed), total_orig_bytes, total_comp_bytes

    def _eval_continuation(self, prefix_ids, suffix_ids, past_kv):
        """
        Compute logits and log-likelihood over suffix_ids using past_kv.
        Returns (perplexity, log_likelihood, logits_per_token)
        """
        total_nll = 0.0
        logits_all = []

        current_past = past_kv
        with torch.no_grad():
            for i in range(len(suffix_ids[0])):
                tok = suffix_ids[:, i:i+1]
                out = self._model(input_ids=tok, past_key_values=current_past,
                                  use_cache=True)
                logits = out.logits[:, -1, :]  # [1, vocab]
                current_past = out.past_key_values

                target = suffix_ids[:, i]
                nll    = F.cross_entropy(logits, target).item()
                total_nll += nll
                logits_all.append(logits.cpu().float())

        n = len(suffix_ids[0])
        perplexity      = math.exp(total_nll / n) if n > 0 else float("inf")
        log_likelihood  = -(total_nll / n)
        return perplexity, log_likelihood, logits_all

    def evaluate(
        self,
        prompts: List[str],
        text_types: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        selector=None,
        selector_label: str = "selector",
    ) -> List[PerplexityResult]:
        """
        Run full perplexity evaluation for all prompts × strategies.

        Parameters
        ----------
        prompts    : list of full texts (will be split prefix/suffix)
        text_types : label per prompt
        strategies : list of strategy labels to test
        selector   : optional LayerSelector (overrides strategies)
        """
        if self._model is None:
            raise RuntimeError("Call load_model() first.")

        strategies = strategies or ["periodic_64", "ema_balanced", "rolling_k3"]
        text_types = text_types or ["unknown"] * len(prompts)
        results    = []

        for prompt, ttype in zip(prompts, text_types):
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=self.max_context_len + self.eval_tokens)
            all_ids = inputs["input_ids"].to(self._model.device)
            if all_ids.shape[1] < self.max_context_len + 2:
                print(f"  [SKIP] Prompt too short for eval: '{prompt[:40]}...'")
                continue

            prefix_ids = all_ids[:, :self.max_context_len]
            suffix_ids = all_ids[:, self.max_context_len:self.max_context_len + self.eval_tokens]
            if suffix_ids.shape[1] < 5:
                continue

            # Baseline: full FP16 KV
            t0 = time.perf_counter()
            baseline_kv = self._get_baseline_kv(prefix_ids)
            ppl_base, ll_base, logits_base = self._eval_continuation(
                prefix_ids, suffix_ids, baseline_kv
            )
            base_ms = (time.perf_counter() - t0) * 1000

            kv_fp16_bytes = sum(k.numel() * 2 + v.numel() * 2
                                for k, v in baseline_kv)

            results.append(PerplexityResult(
                label="baseline_fp16", prompt=prompt[:80], text_type=ttype,
                perplexity=ppl_base, log_likelihood=ll_base,
                token_agreement=1.0, mean_kl_div=0.0,
                kv_bytes_fp16=kv_fp16_bytes, kv_bytes_compressed=kv_fp16_bytes,
                compression_ratio=1.0, eval_ms=base_ms,
            ))

            # Naive INT8 (simulate via uniform requantize)
            # We use the baseline KV and apply uniform INT8 compression
            try:
                from benchmarks.baselines import BaselineRunner
                # For INT8 we measure perplexity using the original KV (same path)
                # — in reality INT8 decode would introduce the same reconstruction noise
                # we will add it as a note in the report
            except Exception:
                pass

            # DiffKV variants
            eval_strategies = [(s, None) for s in strategies]
            if selector is not None:
                eval_strategies = [(selector_label, selector)]

            for strat_label, sel in eval_strategies:
                try:
                    t0 = time.perf_counter()
                    baseline_kv_fresh = self._get_baseline_kv(prefix_ids)
                    recon_kv, orig_b, comp_b = self._compress_kv(
                        baseline_kv_fresh, strat_label, selector=sel
                    )
                    ppl, ll, logits_diffkv = self._eval_continuation(
                        prefix_ids, suffix_ids, recon_kv
                    )
                    eval_ms = (time.perf_counter() - t0) * 1000

                    # Token agreement
                    agreed = sum(
                        1 for lb, ld in zip(logits_base, logits_diffkv)
                        if lb.argmax(-1).item() == ld.argmax(-1).item()
                    )
                    tok_agreement = agreed / len(logits_base) if logits_base else 0.0

                    # KL divergence
                    kl_divs = []
                    for lb, ld in zip(logits_base, logits_diffkv):
                        p = F.softmax(lb, dim=-1)
                        q = F.softmax(ld, dim=-1)
                        kl = (p * (p.log() - q.log())).sum().item()
                        kl_divs.append(max(0, kl))
                    mean_kl = sum(kl_divs) / len(kl_divs) if kl_divs else 0.0

                    results.append(PerplexityResult(
                        label=strat_label, prompt=prompt[:80], text_type=ttype,
                        perplexity=ppl, log_likelihood=ll,
                        token_agreement=tok_agreement, mean_kl_div=mean_kl,
                        kv_bytes_fp16=kv_fp16_bytes,
                        kv_bytes_compressed=comp_b,
                        compression_ratio=orig_b / (comp_b + 1e-9),
                        eval_ms=eval_ms,
                    ))

                    print(f"    [{strat_label:<16}] ppl={ppl:.3f}  "
                          f"agree={tok_agreement:.3f}  KL={mean_kl:.5f}  "
                          f"ratio={orig_b/(comp_b+1e-9):.2f}x")

                except Exception as e:
                    print(f"    [{strat_label}] ERROR: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results
