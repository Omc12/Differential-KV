#!/usr/bin/env python3
"""
Differential-KV (DiffKV) Comprehensive MLSys Research Evaluation Suite.

REWRITE 16.0 — CORRECT MEASUREMENT CORE + FAITHFUL BASELINES + RULER.
================================================================================
This version fixes the two measurement bugs that made REWRITE 15.0 unusable for
a paper, removes several reviewer-fatal confounds, and adds the standard
long-context benchmarks / baselines the field expects in 2025-2026.

WHAT CHANGED vs REWRITE 15.0 (and WHY it matters for the paper)
--------------------------------------------------------------------------------
FIXED — Showstopper 1 (DiffKV memory was always reported as an error):
    15.0 summed tensors from `wrapper.manager.session_blocks`, which is the
    EMPTY legacy dict on CUDA — the streaming path stores blocks in
    `manager._streaming_mgr.session_blocks`. Every low/mid/high run therefore
    returned {"status":"error"} with zero KV memory. We now use the proven
    `analytic_kv_bytes()` core from `run_nat_eval.py`, which reads the corrected
    `mgr.sessions` property AND `native_pool._pool_mb()` for the REAL physical
    pool allocation. The compression ratio we report is dense_equiv / pool
    physical — the honest 1.4-2.5x, not the old ~26x fake counter.

FIXED — Showstopper 2 (DiffKV decode TPS secretly included a 2nd full prefill):
    15.0 measured prefill on a hand-built session `bench_XXXX` but then called
    `wrapper.generate(prompt=full_prompt)`, which runs on session "default",
    CLEARS it, and re-prefills the entire prompt before decoding — so decode_tps
    was gen_len / (full_prefill + decode) and biased massively against DiffKV,
    while the dense branch measured pure incremental decode. We now prefill,
    compress, and decode on ONE session with a token-by-token loop (ported from
    `run_nat_eval.py`) — no `generate()`, no double prefill. The dense/baseline
    branches use the SAME chunk size and the SAME decode loop, so the phase
    timings are like-for-like.

FIXED — Confounds:
    * Weight dtype is now held at FP16 for EVERY method (DiffKV low preset used
      to silently switch weights to 4-bit NF4, so "memory vs preset" conflated
      4-bit weights with KV compression). Quantization is orthogonal/composable
      and is called out as future work, not folded into the KV numbers.
    * ONE consistent memory metric for all methods: kv_physical_gb (real KV
      bytes), kv_dense_equiv_gb (dense KV at that length), compression_ratio,
      plus true peak VRAM per phase (weights identical, so peak VRAM is fair).
    * exp4 (tradeoff) now pairs memory and quality at the SAME context length.
    * exp12/exp13 no longer relabel the same 3 academic papers as "Legal",
      "10-K", "Fiction". Honest corpus names; a hook to plug real LongBench/RULER
      datasets is documented.
    * Quality/recall is now multi-sample with Wilson 95% CIs, not single-shot.
    * nsys/ncu parsing matches actual Nsight CSV columns and is best-effort.

ADDED — Faithful baselines (through the identical harness):
    Dense FP16 | INT8-KV (per-token) | KIVI-style 2/3/4-bit (per-channel K,
    per-token V) | StreamingLLM (attention sink + recency) | SnapKV (REAL
    accumulated-attention eviction) | KeyNorm-HH (H2O-style key-norm proxy,
    labeled honestly) | DiffKV (ours).
    SnapKV is faithful: it ranks prefix tokens by real attention mass from a
    small observation window, materializing only the [B,H,window,S] map (never
    the infeasible [H,S,S]), via a model loaded with attn_implementation="eager".
    Its selection math is CPU-tested; the output_attentions integration needs GPU
    validation. KeyNorm-HH is kept as the cheaper key-norm proxy for contrast.

ADDED — The two experiments that decide the paper:
    exp22 (accuracy/KV-memory FRONTIER at fixed long context): sweeps every
    method across its memory dial (DiffKV presets, KIVI 2/3/4-bit, SnapKV/HH/
    Streaming budgets). DiffKV is a contribution ONLY if it is un-dominated —
    matches dense quality at a budget where the strong baselines have degraded.
    exp23 (quality-vs-context curve): recall vs context for dense/DiffKV/KIVI/
    SnapKV/Streaming; the claim holds iff DiffKV tracks dense out to long context
    while the others peel away. Run these TWO first — they are cheap and tell you
    whether the full grid is worth the A100 hours.

PERF — The prefill-compress lever (the conceded axis) lives in a SEPARATE
    standalone decision test, not this suite: colab/gram_eigh_decision.py. It
    CPU-verifies Gram-eigh ≡ SVD, then (with --gpu-ab) A/Bs compress time + NIAH
    recall for baseline SVD vs Gram-eigh vs the r_proj<=32 recipe, and prints
    whether it is safe to make Gram-eigh the default. The config['extra_env']
    passthrough below (per-call DIFFKV_* overrides) is what that script uses.
    Everything else about decode/prefill speed: measure from this harness first.

ADDED — RULER-style long-context suite (synthetic, dataset-free): single-key,
    multi-key, multi-value, multi-query NIAH, variable tracking, and frequent-
    word extraction, each with recall + Wilson CI. Single-needle NIAH alone is
    considered too easy now; RULER is the current standard.

ADDED — Accuracy-vs-memory Pareto emission and optional second model family
    (e.g. Llama-3.1-8B) so claims are not Qwen-specific.

CAVEAT: This Mac cannot run CUDA. The non-GPU logic (RULER generation, metrics,
    Wilson CI, baseline kernels) is CPU-smoke-tested; the CUDA measurement core
    is ported line-for-line from the GPU-validated `run_nat_eval.py`. Validate on
    the A100 before quoting numbers.
"""

import os
import sys
import gc
import re
import csv
import json
import time
import math
import random
import argparse
import subprocess
import traceback
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

# Enable PyTorch CUDA Allocator expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("DIFFKV_DIAG", "0")

# Setup Repository Paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
BENCHMARKS = os.path.join(REPO, "benchmarks")

if ACTIVE not in sys.path:
    sys.path.insert(0, ACTIVE)
if BENCHMARKS not in sys.path:
    sys.path.insert(0, BENCHMARKS)

import torch

# SSL patch for downloading HuggingFace models in restricted network environments
import ssl
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Standard Public Corpora Paths
NAT_PAPER_PATH = os.path.join(ACTIVE, "nat_paper.txt")
BERRY_PAPER_PATH = os.path.join(BENCHMARKS, "berry_paper.txt")
RANDOM_PAPER_PATH = os.path.join(BENCHMARKS, "random_features_paper.txt")

# Methods that run on a plain HuggingFace model + a post-prefill KV transform.
# Everything else ("low"/"mid"/"high" and the adaptive_* aliases) is DiffKV and
# runs through PyTorchDiffKVHFWrapper.
DENSE_FAMILY_METHODS = ("dense", "int8_kv", "kivi2", "streaming", "keynorm_hh", "snapkv")

# Adaptive presets map onto a base DiffKV preset + extra env flags.
ADAPTIVE_PRESETS = {
    "adaptive_rank": ("low", {"DIFFKV_LAYER_ADAPTIVE_RANK": "1"}),
    "adaptive_stream": ("low", {"DIFFKV_LAYER_ADAPTIVE_RANK": "1", "DIFFKV_STREAMING_COMPRESS": "1"}),
}


def load_file_text(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return "Sample context text for benchmark context evaluation."


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Helpers — latency CIs (t-style normal approx) + Wilson CI (recall)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "ci95_margin": 0.0}
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return {"mean": m, "std": 0.0, "ci95_low": m, "ci95_high": m, "ci95_margin": 0.0}
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    margin = 1.96 * (std / math.sqrt(n))
    return {"mean": m, "std": std, "ci95_low": m - margin, "ci95_high": m + margin, "ci95_margin": margin}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Dict[str, float]:
    """Wilson score 95% CI for a binomial proportion — the correct interval for
    pass/fail recall over a finite number of samples (normal-approx CIs are wrong
    near 0% / 100%, exactly where recall lives)."""
    if n == 0:
        return {"p": 0.0, "low": 0.0, "high": 0.0, "margin": 0.0, "n": 0}
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    low, high = max(0.0, center - half), min(1.0, center + half)
    return {"p": p * 100.0, "low": low * 100.0, "high": high * 100.0,
            "margin": (high - low) / 2 * 100.0, "n": n}


# ─────────────────────────────────────────────────────────────────────────────
# NLP Metrics
# ─────────────────────────────────────────────────────────────────────────────

def exact_match_score(predicted: str, ground_truth: str) -> float:
    p = predicted.strip().lower()
    gt = ground_truth.strip().lower()
    if not gt:
        return 0.0
    return 100.0 if gt in p else 0.0


def token_f1_score(predicted: str, ground_truth: str) -> float:
    pred_tokens = predicted.strip().lower().split()
    gt_tokens = ground_truth.strip().lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return ((2 * precision * recall) / (precision + recall)) * 100.0


def concept_synonym_recall(predicted: str, key_concepts: List[str]) -> float:
    p = predicted.strip().lower()
    found = 0
    for concept in key_concepts:
        terms = [t.strip().lower() for t in concept.split("|")]
        if any(term in p for term in terms):
            found += 1
    return (found / len(key_concepts)) * 100.0 if key_concepts else 0.0


def answer_set_recall(predicted: str, answers: List[str]) -> float:
    """Fraction of the required answer strings that appear (case-insensitive) in
    the output. Used by the multi-value / multi-query RULER tasks."""
    if not answers:
        return 0.0
    p = predicted.upper()
    found = sum(1 for a in answers if a.upper() in p)
    return found / len(answers)


# ─────────────────────────────────────────────────────────────────────────────
# Random passcode needle generator (single-needle NIAH)
# ─────────────────────────────────────────────────────────────────────────────

_PREFIXES = ["OMEGA", "SIGMA", "THETA", "LAMBDA", "KAPPA", "NEXUS", "CYPHER", "VORTEX", "APEX", "TITAN"]
_SUFFIXES = ["DELTA", "BETA", "ALPHA", "GAMMA", "ZETA", "PRIME", "MATRIX", "VECTOR", "SHIELD", "ORBIT"]


def _rand_code(rng: random.Random) -> str:
    return f"{rng.choice(_PREFIXES)}-{rng.randint(1000, 9999)}-{rng.choice(_SUFFIXES)}"


def generate_random_needles(count: int = 50) -> List[Tuple[str, str]]:
    rng = random.Random(42)
    needles, seen = [], set()
    while len(needles) < count:
        code = _rand_code(rng)
        if code not in seen:
            seen.add(code)
            needles.append((code, f"The secret security passcode is {code}."))
    return needles


# ─────────────────────────────────────────────────────────────────────────────
# RULER-style synthetic long-context tasks (dataset-free, spirit of RULER).
#
# Each generator returns a dict:
#   {"context": str, "question": str, "answers": [str, ...], "mode": <match>}
# where <match> is "substring" (answers[0] must appear) or "set" (recall over
# the full answer set). Filler is real prose (nat_paper) so the retrieval target
# is genuinely buried, not trivially separable from random tokens.
# ─────────────────────────────────────────────────────────────────────────────

def _filler_units(tokenizer, target_tokens: int) -> List[str]:
    """Return a list of sentence-ish filler chunks whose total length ~ target."""
    text = load_file_text(NAT_PAPER_PATH)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    if not sents:
        sents = ["This is neutral filler text used to pad the context window."]
    # Estimate tokens/sentence once to know how many sentences to emit.
    sample = " ".join(sents[:20]) or sents[0]
    tok_per_char = max(1e-3, len(tokenizer.encode(sample, add_special_tokens=False)) / max(1, len(sample)))
    out, acc = [], 0
    i = 0
    while acc < target_tokens:
        s = sents[i % len(sents)]
        out.append(s)
        acc += int(len(s) * tok_per_char) + 1
        i += 1
    return out


def _weave(units: List[str], inserts: List[Tuple[float, str]]) -> str:
    """Insert (depth, sentence) items into the filler at fractional depths."""
    units = list(units)
    for depth, sent in sorted(inserts, key=lambda x: x[0]):
        idx = min(len(units), max(0, int(len(units) * depth)))
        units.insert(idx, sent)
    return "\n".join(units)


def ruler_task(name: str, tokenizer, ctx_len: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    filler = _filler_units(tokenizer, ctx_len)

    if name == "niah_single":
        code = _rand_code(rng)
        ctx = _weave(filler, [(rng.uniform(0.15, 0.85),
                               f"The magic passcode for the vault is {code}.")])
        return {"context": ctx, "question": "What is the magic passcode for the vault? State the code exactly.",
                "answers": [code], "mode": "substring"}

    if name == "niah_multikey":
        # Several distinct keyed needles; retrieve the value for ONE key. The
        # distractor needles share the exact sentence template (only key/value
        # differ), so the model cannot cheat on surface form.
        k = 4
        keys = [f"K{rng.randint(100, 999)}" for _ in range(k)]
        vals = [_rand_code(rng) for _ in range(k)]
        inserts = [(rng.uniform(0.1, 0.9), f"The access code for account {keys[i]} is {vals[i]}.")
                   for i in range(k)]
        ctx = _weave(filler, inserts)
        t = rng.randrange(k)
        return {"context": ctx, "question": f"What is the access code for account {keys[t]}? State it exactly.",
                "answers": [vals[t]], "mode": "substring"}

    if name == "niah_multivalue":
        # One key, several values — retrieve ALL of them (set recall).
        key = f"K{rng.randint(100, 999)}"
        vals = [_rand_code(rng) for _ in range(3)]
        inserts = [(rng.uniform(0.1, 0.9), f"Account {key} has authorized code {v}.") for v in vals]
        ctx = _weave(filler, inserts)
        return {"context": ctx, "question": f"List every authorized code for account {key}.",
                "answers": vals, "mode": "set"}

    if name == "niah_multiquery":
        # Several keys; retrieve values for a queried subset (set recall).
        k = 4
        keys = [f"K{rng.randint(100, 999)}" for _ in range(k)]
        vals = [_rand_code(rng) for _ in range(k)]
        inserts = [(rng.uniform(0.1, 0.9), f"The token for slot {keys[i]} is {vals[i]}.") for i in range(k)]
        ctx = _weave(filler, inserts)
        q = sorted(rng.sample(range(k), 2))
        want = [vals[i] for i in q]
        wanted_keys = ", ".join(keys[i] for i in q)
        return {"context": ctx, "question": f"State the tokens for slots {wanted_keys}, exactly.",
                "answers": want, "mode": "set"}

    if name == "vt":
        # Variable tracking: a chain of assignments; report the final value.
        depth = 5
        base = rng.randint(10000, 99999)
        names = [f"VAR{rng.randint(10, 99)}_{i}" for i in range(depth)]
        lines = [f"{names[0]} = {base}."]
        for i in range(1, depth):
            lines.append(f"{names[i]} = {names[i-1]}.")
        inserts = [(rng.uniform(0.1, 0.9), ln) for ln in lines]
        ctx = _weave(filler, inserts)
        return {"context": ctx, "question": f"Following the assignments, what numeric value does {names[-1]} hold?",
                "answers": [str(base)], "mode": "substring"}

    if name == "fwe":
        # Frequent-word extraction: a special token repeated far more often than
        # decoys; report it. (RULER's aggregation family.)
        target = f"ZQX{rng.randint(100, 999)}"
        decoys = [f"ZQX{rng.randint(100, 999)}" for _ in range(3)]
        inserts = []
        for _ in range(12):
            inserts.append((rng.uniform(0.05, 0.95), f"Observation marker {target} recorded."))
        for d in decoys:
            inserts.append((rng.uniform(0.05, 0.95), f"Observation marker {d} recorded."))
        ctx = _weave(filler, inserts)
        return {"context": ctx, "question": "Which observation marker (ZQX...) appears most frequently? State it exactly.",
                "answers": [target], "mode": "substring"}

    raise ValueError(f"unknown ruler task {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Baseline KV-transform kernels (operate on HF past_key_values after prefill).
# Each returns (K_out, V_out, physical_bytes_gb) for a single layer.
# ─────────────────────────────────────────────────────────────────────────────

def apply_int8_kv_quantization(key_states: torch.Tensor, value_states: torch.Tensor):
    """Per-token symmetric INT8 KV quantization (standard 8-bit KV baseline)."""
    k_scale = torch.clamp(torch.max(torch.abs(key_states), dim=-1, keepdim=True).values / 127.0, min=1e-8)
    v_scale = torch.clamp(torch.max(torch.abs(value_states), dim=-1, keepdim=True).values / 127.0, min=1e-8)
    k_int8 = torch.clamp(torch.round(key_states / k_scale), -128, 127).to(torch.int8)
    v_int8 = torch.clamp(torch.round(value_states / v_scale), -128, 127).to(torch.int8)
    k_dq = (k_int8.to(torch.float16) * k_scale).to(key_states.dtype)
    v_dq = (v_int8.to(torch.float16) * v_scale).to(value_states.dtype)
    bytes_phys = (k_int8.numel() + v_int8.numel()) * 1.0 + (k_scale.numel() + v_scale.numel()) * 4.0
    return k_dq, v_dq, bytes_phys / 1e9


def _quant_group(x: torch.Tensor, bits: int, dim: int):
    """Asymmetric group-wise quant of x along `dim` to `bits`. Returns dequant
    tensor + (bytes for packed codes + fp16 scale + fp16 zero)."""
    qmax = (1 << bits) - 1
    xmin = torch.amin(x, dim=dim, keepdim=True)
    xmax = torch.amax(x, dim=dim, keepdim=True)
    scale = torch.clamp((xmax - xmin) / qmax, min=1e-8)
    q = torch.clamp(torch.round((x - xmin) / scale), 0, qmax)
    dq = (q * scale + xmin).to(x.dtype)
    n_groups = 1
    for d in range(x.dim()):
        if d != dim:
            n_groups *= x.shape[d]
    bytes_codes = x.numel() * bits / 8.0
    bytes_meta = n_groups * 2 * 2.0  # fp16 scale + fp16 zero per group
    return dq, bytes_codes + bytes_meta


def apply_kivi_2bit(key_states: torch.Tensor, value_states: torch.Tensor, bits: int = 2):
    """KIVI-style low-bit KV quantization: Key quantized PER-CHANNEL (along the
    token axis, which tames per-channel outliers), Value quantized PER-TOKEN
    (along the feature axis). `bits` defaults to 2 (the headline KIVI setting);
    the memory/quality frontier sweeps bits in {2,3,4}.
    Shapes: [B, H, S, D] — token axis = 2, feature axis = 3."""
    k_dq, k_bytes = _quant_group(key_states, bits=bits, dim=2)    # per-channel (over tokens)
    v_dq, v_bytes = _quant_group(value_states, bits=bits, dim=3)  # per-token (over features)
    return k_dq, v_dq, (k_bytes + v_bytes) / 1e9


def apply_streaming_llm(key_states: torch.Tensor, value_states: torch.Tensor,
                        n_sink: int = 4, recency_window: int = 256):
    """StreamingLLM: keep the first `n_sink` attention-sink tokens + the last
    `recency_window` tokens, drop the middle. Faithful, needs no attention
    scores. Absolute RoPE positions are preserved on the kept tokens."""
    seq_len = key_states.shape[2]
    if seq_len <= n_sink + recency_window:
        b = (key_states.numel() + value_states.numel()) * 2.0 / 1e9
        return key_states, value_states, b
    k = torch.cat([key_states[:, :, :n_sink, :], key_states[:, :, -recency_window:, :]], dim=2)
    v = torch.cat([value_states[:, :, :n_sink, :], value_states[:, :, -recency_window:, :]], dim=2)
    return k, v, (k.numel() + v.numel()) * 2.0 / 1e9


def apply_keynorm_hh(key_states: torch.Tensor, value_states: torch.Tensor,
                     heavy_hitter_budget: int = 128, recency_window: int = 128):
    """H2O-STYLE heavy-hitter eviction using KEY L2-NORM as the importance proxy.

    HONESTY NOTE: true H2O (Zhang et al.) and SnapKV rank history tokens by
    ACCUMULATED ATTENTION MASS from an observation window, not key norm. Getting
    real attention scores needs an attention-observation hook (materializing full
    [H,S,S] attention for all layers is infeasible at long context). Key-norm is
    a documented, cheaper importance proxy that correlates with attention mass;
    we label this baseline "KeyNorm-HH (H2O-style)" rather than claim it is exact
    H2O. A faithful attention-hook variant is left as an extension. StreamingLLM
    and KIVI-2bit (both faithful) are the primary standard baselines here."""
    seq_len = key_states.shape[2]
    total_budget = heavy_hitter_budget + recency_window
    if seq_len <= total_budget:
        b = (key_states.numel() + value_states.numel()) * 2.0 / 1e9
        return key_states, value_states, b
    recent_k = key_states[:, :, -recency_window:, :]
    recent_v = value_states[:, :, -recency_window:, :]
    hist_k = key_states[:, :, :-recency_window, :]
    hist_v = value_states[:, :, :-recency_window, :]
    scores = torch.norm(hist_k, dim=-1).mean(dim=1)  # [B, S_hist]
    top = torch.topk(scores, k=heavy_hitter_budget, dim=-1).indices[0].sort().values
    k = torch.cat([hist_k[:, :, top, :], recent_k], dim=2)
    v = torch.cat([hist_v[:, :, top, :], recent_v], dim=2)
    return k, v, (k.numel() + v.numel()) * 2.0 / 1e9


_BASELINE_KERNELS = {
    "int8_kv": apply_int8_kv_quantization,
    "kivi2": apply_kivi_2bit,
    "streaming": apply_streaming_llm,
    "keynorm_hh": apply_keynorm_hh,
}


# ─────────────────────────────────────────────────────────────────────────────
# DiffKV physical-memory accounting — ported from run_nat_eval.py::analytic_kv_bytes
# (the GPU-validated version). Reads the CORRECTED mgr.sessions property and the
# real pool allocation; does NOT touch the empty legacy `manager.session_blocks`.
# ─────────────────────────────────────────────────────────────────────────────

def analytic_kv_bytes(mgr, seq_len: int, sid: str) -> Dict[str, float]:
    L = mgr.num_layers
    Hkv = mgr.kv_heads
    d = mgr.head_dim
    fp16 = 2
    B = int(getattr(mgr, "micro_block_size", 0) or getattr(mgr, "block_size", 64))
    pool = getattr(mgr, "native_pool", None)
    r = int(getattr(pool, "rank", None) or mgr.rank)

    kv_tok = Hkv * d * fp16 * 2
    lowrank_block = (B * r * 1 + 2 * Hkv * r * d * fp16 + 2 * Hkv * d * fp16 + 8)

    s0 = mgr.sessions.get(sid)
    nb = s0["num_blocks"][0] if s0 else 0
    dl = s0["dense_lens"][0] if s0 else 0
    res_n0 = s0["comp_res_n"][0][:nb] if s0 else []
    res_tokens_used = int(sum(res_n0))

    store_used = L * (nb * lowrank_block + res_tokens_used * kv_tok + dl * kv_tok)
    pool_physical = 0
    if pool is not None and hasattr(pool, "_pool_mb"):
        pool_physical = int(pool._pool_mb() * 1024 ** 2)
    dense_equiv = L * seq_len * kv_tok
    return {
        "store_used_bytes": store_used,
        "pool_physical_bytes": pool_physical,
        "dense_equiv_bytes": dense_equiv,
        "blocks_layer0": nb,
        "residual_tokens_layer0": res_tokens_used,
        "dense_window_tokens": dl,
    }


def wait_for_compression(mgr, session_id: str) -> bool:
    streaming_mgr = getattr(mgr, "_streaming_mgr", None)
    if streaming_mgr is None:
        return True
    timeout_s = float(os.environ.get("DIFFKV_COMPRESSION_TIMEOUT_S", "30"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hasattr(mgr, "finalize_compressed_blocks"):
            mgr.finalize_compressed_blocks()
        blocks = streaming_mgr.session_blocks.get(session_id, {})
        pending = sum(1 for lb in blocks.values() for b in lb
                      if getattr(b, "state", None) in ("SUBMITTED", "CPU_COMPRESSED"))
        if pending == 0:
            return True
        time.sleep(0.002)
    print(f"[bench] WARNING: compression barrier timed out for {session_id}", flush=True)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Prompt building (chat-template-aware, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, context: str, question: str,
                 system: str = "You are a helpful assistant. Answer strictly using the provided context.") -> str:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system}\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAssistant:"


def _fit_context(tokenizer, context: str, ctx_len: int) -> str:
    """Repeat/truncate `context` to ~ctx_len tokens (leaving headroom for the
    chat template + question)."""
    c_tokens = tokenizer.encode(context, add_special_tokens=False)
    if not c_tokens:
        return context
    reps = max(1, (ctx_len // len(c_tokens)) + 1)
    target = (c_tokens * reps)[:max(64, ctx_len - 96)]
    return tokenizer.decode(target)


def _derive_stop_ids(tokenizer) -> set:
    stop = set()
    for sid in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(sid, list):
            stop.update(sid)
        elif isinstance(sid, int):
            stop.add(sid)
    for w in ["<|im_end|>", "<|end_of_text|>", "<|eot_id|>", "</s>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(w)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop.add(tid)
    return stop


# ─────────────────────────────────────────────────────────────────────────────
# HF cache helpers (support both legacy tuple cache and DynamicCache)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_to_legacy(pkv):
    if pkv is None:
        return None
    if hasattr(pkv, "to_legacy_cache"):
        try:
            return pkv.to_legacy_cache()
        except Exception:
            return pkv
    return pkv


def _cache_from_legacy(legacy):
    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


# ─────────────────────────────────────────────────────────────────────────────
# Single-trial runners
# ─────────────────────────────────────────────────────────────────────────────

def _chunked_prefill_dense(model, ids: List[int], device: str, CH: int):
    """Chunked dense prefill returning (past_key_values, last_logits_gpu)."""
    past = None
    out = None
    for cs in range(0, len(ids), CH):
        ch = ids[cs:cs + CH]
        pos = torch.tensor([list(range(cs, cs + len(ch)))], device=device)
        out = model(input_ids=torch.tensor([ch], device=device),
                    position_ids=pos, past_key_values=past, use_cache=True)
        past = out.past_key_values
    return past, out.logits[0, -1].float()


def _dense_family_trial(model, tokenizer, ids: List[int], method: str, device: str,
                        gen_len: int, stop_ids: set, CH: int,
                        method_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One prefill+transform+decode trial for a dense-family method.
    method_params overrides the KV-transform knobs (e.g. {"bits": 4} for KIVI,
    {"recency_window": 1024} for StreamingLLM) — used by the memory/quality
    frontier sweep."""
    method_params = method_params or {}
    prompt_len = len(ids)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Prefill (forward) ──
    t0 = time.perf_counter()
    with torch.no_grad():
        past, last_logits = _chunked_prefill_dense(model, ids, device, CH)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_s = time.perf_counter() - t0
    peak_prefill = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # Dense KV bytes (logical + reference) — identical formula for every method.
    L = model.config.num_hidden_layers
    Hkv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    d = model.config.hidden_size // model.config.num_attention_heads
    dense_kv_bytes = (L * prompt_len * Hkv * d * 2 * 2) / 1e9
    phys_bytes = dense_kv_bytes

    # ── KV transform (baseline "compression" stage) ──
    t1 = time.perf_counter()
    if method in _BASELINE_KERNELS:
        legacy = _cache_to_legacy(past)
        new_layers, total = [], 0.0
        for (k, v) in legacy:
            k2, v2, gb = _BASELINE_KERNELS[method](k, v, **method_params)
            new_layers.append((k2, v2))
            total += gb
        past = _cache_from_legacy(tuple(new_layers))
        phys_bytes = total
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    comp_s = time.perf_counter() - t1

    # ── Decode ──
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cur = prompt_len
    gen_ids = []
    _inp = torch.zeros((1, 1), dtype=torch.long, device=device)
    _pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    t2 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len):
            nid = int(torch.argmax(last_logits).item())
            if nid in stop_ids:
                break
            gen_ids.append(nid)
            _inp[0, 0] = nid
            _pos[0, 0] = cur
            out = model(input_ids=_inp, position_ids=_pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[0, -1].float()
            cur += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dec_s = time.perf_counter() - t2
    peak_decode = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    # HONESTY NOTE on kv_physical_gb for the quant baselines (int8_kv, kivi2):
    # this is the ALGORITHMIC KV footprint (bytes the method would store). The
    # harness dequantizes to fp16 for the decode forward (no fused quant-attention
    # kernel), so peak_decode_vram_gb reflects fp16, not the quantized bytes —
    # i.e. kv_physical_gb is the *theoretical* KV memory for quant methods, while
    # for DiffKV it is the *realized* pool allocation and for dense/eviction it is
    # the realized fp16 KV. Report kv_physical_gb as the KV-footprint axis (the
    # standard axis in KV-compression papers); use peak VRAM only where the method
    # actually realizes its footprint (dense, eviction, DiffKV).
    return {
        "prefill_forward_s": fwd_s, "prefill_compress_s": comp_s, "decode_time_s": dec_s,
        "decode_tps": len(gen_ids) / dec_s if dec_s > 0 else 0.0,
        "peak_prefill_vram_gb": peak_prefill, "peak_decode_vram_gb": peak_decode,
        "kv_physical_gb": phys_bytes, "kv_dense_equiv_gb": dense_kv_bytes,
        "kv_footprint_realized": method in ("dense", "streaming", "keynorm_hh"),
        "gen_len": len(gen_ids), "output_text": tokenizer.decode(gen_ids),
    }


def _diffkv_trial(w, ids: List[int], device: str, gen_len: int, stop_ids: set) -> Dict[str, Any]:
    """One prefill+compress+decode trial on a SINGLE DiffKV session — no
    generate(), no double prefill. Ported from run_nat_eval.py."""
    mgr, model = w.manager, w.model
    prompt_len = len(ids)
    sid = f"bench_{random.randint(10**6, 10**7)}"
    mgr.clear_session(sid)
    if not hasattr(w, "_session_token_ids"):
        w._session_token_ids = {}
    w._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=prompt_len)
    if hasattr(mgr, "register_prefill_tokens"):
        mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
    model._diffkv_session_ids = [sid]
    if hasattr(w, "_cuda_graph_runner") and getattr(w, "_cuda_graph_runner") is not None:
        try:
            w._cuda_graph_runner.invalidate()
        except Exception:
            pass

    # Chunk size = active preset's prefill_chunk_size, rounded to block capacity.
    CH = int(getattr(getattr(mgr, "config", None), "prefill_chunk_size", 1024) or 1024)
    if torch.cuda.is_available() and hasattr(mgr, "get_session_micro_block_size"):
        _mbs = mgr.get_session_micro_block_size(sid)
        cap = max(2, int(_mbs) + 1)
        CH = ((CH + cap - 1) // cap) * cap

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Prefill forward (exact causal; SVD deferred to boundary) ──
    t0 = time.perf_counter()
    out = None
    with torch.no_grad():
        for cs in range(0, len(ids), CH):
            ch = ids[cs:cs + CH]
            if hasattr(mgr, "finalize_compressed_blocks"):
                mgr.finalize_compressed_blocks()
            out = model(input_ids=torch.tensor([ch], device=device),
                        position_ids=torch.tensor([list(range(cs, cs + len(ch)))], device=device),
                        use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_s = time.perf_counter() - t0
    peak_prefill = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    last_logits = out.logits[0, -1].float().clone()

    # ── Compression (SVD publish + SRL index) ──
    t1 = time.perf_counter()
    with torch.no_grad():
        if hasattr(mgr, "compress_deferred_prefill_blocks"):
            mgr.compress_deferred_prefill_blocks(sid)
        wait_for_compression(mgr, sid)
        if hasattr(mgr, "finalize_srl_index"):
            mgr.finalize_srl_index(sid, cached_len=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    comp_s = time.perf_counter() - t1

    kv = analytic_kv_bytes(mgr, prompt_len, sid)

    # ── Decode (token-by-token on the SAME session) ──
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cur = prompt_len
    gen_ids = []
    _inp = torch.zeros((1, 1), dtype=torch.long, device=device)
    _pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    t2 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len):
            nid = int(torch.argmax(last_logits).item())
            if nid in stop_ids:
                break
            gen_ids.append(nid)
            _inp[0, 0] = nid
            _pos[0, 0] = cur
            out = model(input_ids=_inp, position_ids=_pos, use_cache=True)
            last_logits = out.logits[0, -1].float()
            cur += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dec_s = time.perf_counter() - t2
    peak_decode = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    text = w.tokenizer.decode(gen_ids)
    pool_phys = kv["pool_physical_bytes"] / 1e9
    dense_eq = kv["dense_equiv_bytes"] / 1e9
    mgr.clear_session(sid)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "prefill_forward_s": fwd_s, "prefill_compress_s": comp_s, "decode_time_s": dec_s,
        "decode_tps": len(gen_ids) / dec_s if dec_s > 0 else 0.0,
        "peak_prefill_vram_gb": peak_prefill, "peak_decode_vram_gb": peak_decode,
        "kv_physical_gb": pool_phys, "kv_dense_equiv_gb": dense_eq,
        "kv_logical_gb": kv["store_used_bytes"] / 1e9, "kv_blocks_layer0": kv["blocks_layer0"],
        "gen_len": len(gen_ids), "output_text": text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Faithful SnapKV (attention-observation eviction) — the strong eviction baseline.
#
# Unlike the key-norm proxy, this ranks prefix tokens by REAL accumulated
# attention mass from an observation window (the last `window` prompt tokens),
# exactly as SnapKV (Li et al. 2024) prescribes. It is feasible because we only
# ever materialize the [B,H,window,S] observation-window attention (window≈32),
# never the full [B,H,S,S] map. Requires the model loaded with
# attn_implementation="eager" so HF returns attention weights.
# GPU-VALIDATE before quoting: the pure selection math is CPU-tested; the
# output_attentions integration depends on the transformers build.
# ─────────────────────────────────────────────────────────────────────────────

def _snapkv_select(prefix_scores: torch.Tensor, keep: int, pool_kernel: int = 7) -> torch.Tensor:
    """prefix_scores [B,H,P] accumulated attention per prefix position → indices
    [B,H,keep] of the tokens to retain. A 1-D max-pool (SnapKV's clustering step)
    keeps contiguous high-attention spans rather than isolated spikes."""
    import torch.nn.functional as F
    P = prefix_scores.shape[-1]
    keep = max(1, min(keep, P))
    s = prefix_scores
    if pool_kernel and pool_kernel > 1:
        s = F.max_pool1d(prefix_scores, kernel_size=pool_kernel, stride=1, padding=pool_kernel // 2)
        s = s[..., :P]
    idx = torch.topk(s, k=keep, dim=-1).indices
    return idx.sort(dim=-1).values


def _snapkv_trial(model, tokenizer, ids: List[int], device: str, gen_len: int, stop_ids: set,
                  CH: int, budget: int = 512, window: int = 32, pool_kernel: int = 7) -> Dict[str, Any]:
    prompt_len = len(ids)
    window = max(1, min(window, prompt_len // 4 if prompt_len >= 8 else 1))
    prefix_ids = ids[:prompt_len - window]
    obs_ids = ids[prompt_len - window:]
    plen = len(prefix_ids)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Prefill prefix, then score the observation window (real attention) ──
    t0 = time.perf_counter()
    past, _ = _chunked_prefill_dense(model, prefix_ids, device, CH)
    pos = torch.tensor([list(range(plen, prompt_len))], device=device)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([obs_ids], device=device), position_ids=pos,
                    past_key_values=past, use_cache=True, output_attentions=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_s = time.perf_counter() - t0
    peak_prefill = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    last_logits = out.logits[0, -1].float()

    # ── Build the compressed cache: top-`budget` prefix tokens (per head) by
    #    accumulated attention + the always-kept observation window ──
    t1 = time.perf_counter()
    legacy = _cache_to_legacy(out.past_key_values)
    L = len(legacy)
    Hkv, d = legacy[0][0].shape[1], legacy[0][0].shape[3]
    keep = min(budget, plen)
    new_layers, total_bytes = [], 0.0
    for l, (K, V) in enumerate(legacy):
        A = out.attentions[l]                                       # [B, n_q, W, prompt_len]
        prefix_scores = A[..., :plen].to(torch.float32).sum(dim=2)  # [B, n_q, plen]
        # GQA: attention is per QUERY head but the KV cache is per KV head. Pool
        # the query-head scores within each KV group so selection indexes the
        # kv-head cache correctly (Qwen2.5 / Llama-3.1 are grouped-query).
        n_q = prefix_scores.shape[1]
        n_kv = K.shape[1]
        if n_q != n_kv and n_q % n_kv == 0:
            g = n_q // n_kv
            prefix_scores = prefix_scores.view(prefix_scores.shape[0], n_kv, g, plen).mean(dim=2)
        idx = _snapkv_select(prefix_scores, keep, pool_kernel)      # [B, n_kv, keep]
        gi = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        Kp = torch.gather(K[:, :, :plen, :], 2, gi)
        Vp = torch.gather(V[:, :, :plen, :], 2, gi)
        Kc = torch.cat([Kp, K[:, :, plen:, :]], dim=2)
        Vc = torch.cat([Vp, V[:, :, plen:, :]], dim=2)
        new_layers.append((Kc, Vc))
        total_bytes += (Kc.numel() + Vc.numel()) * 2.0
    past = _cache_from_legacy(tuple(new_layers))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    comp_s = time.perf_counter() - t1
    phys_bytes = total_bytes / 1e9
    dense_kv_bytes = (L * prompt_len * Hkv * d * 2 * 2) / 1e9

    # ── Decode ──
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cur = prompt_len
    gen_ids = []
    _inp = torch.zeros((1, 1), dtype=torch.long, device=device)
    _pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    t2 = time.perf_counter()
    with torch.no_grad():
        for _ in range(gen_len):
            nid = int(torch.argmax(last_logits).item())
            if nid in stop_ids:
                break
            gen_ids.append(nid)
            _inp[0, 0] = nid
            _pos[0, 0] = cur
            out = model(input_ids=_inp, position_ids=_pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[0, -1].float()
            cur += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dec_s = time.perf_counter() - t2
    peak_decode = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return {
        "prefill_forward_s": fwd_s, "prefill_compress_s": comp_s, "decode_time_s": dec_s,
        "decode_tps": len(gen_ids) / dec_s if dec_s > 0 else 0.0,
        "peak_prefill_vram_gb": peak_prefill, "peak_decode_vram_gb": peak_decode,
        "kv_physical_gb": phys_bytes, "kv_dense_equiv_gb": dense_kv_bytes,
        "kv_footprint_realized": True, "gen_len": len(gen_ids), "output_text": tokenizer.decode(gen_ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker: build the task prompt, load the right model, run N trials, aggregate.
# ─────────────────────────────────────────────────────────────────────────────

def _build_task(task_type: str, config: Dict[str, Any], tokenizer):
    """Returns (full_prompt, ground_truth, answers, match_mode)."""
    ctx_len = config.get("ctx_len", 4096)

    if task_type == "niah":
        depth = config.get("depth", 0.5)
        code = config.get("needle_code", "OMEGA-7741-DELTA")
        sent = config.get("needle_sent", f"The secret security passcode is {code}.")
        question = "What is the secret security passcode? State the code exactly."
        filler_units = _filler_units(tokenizer, ctx_len)
        # Use the woven context AS-IS: it is already ~ctx_len and holds the needle
        # at the requested depth. Do NOT _fit_context() it — repeating would
        # duplicate the needle (false pass) and truncating a depth-0.9 needle
        # would drop it (false fail). prompt_len is reported exactly downstream.
        ctx = _weave(filler_units, [(depth, sent)])
        return build_prompt(tokenizer, ctx, question), code, [code], "substring"

    if task_type == "ruler":
        # Same rule as NIAH: the RULER context is pre-sized and holds the
        # needles/assignments at fixed depths — never repeat or truncate it.
        spec = ruler_task(config["ruler_task"], tokenizer, ctx_len, config.get("seed", 0))
        gt = spec["answers"][0] if spec["answers"] else ""
        return build_prompt(tokenizer, spec["context"], spec["question"]), gt, spec["answers"], spec["mode"]

    if task_type in ("quality", "doc_domain", "category", "multidoc"):
        context = config.get("context_text", load_file_text(NAT_PAPER_PATH))
        question = config.get("question", "What is the primary topic discussed in the text?")
        gt = config.get("ground_truth", "")
        ctx = _fit_context(tokenizer, context, ctx_len)
        return build_prompt(tokenizer, ctx, question), gt, [gt] if gt else [], "substring"

    # default: summarize
    ctx = _fit_context(tokenizer, load_file_text(NAT_PAPER_PATH), ctx_len)
    return build_prompt(tokenizer, ctx, "Summarize the text."), "", [], "substring"


def _set_diffkv_env(preset: str):
    """Resolve preset / adaptive aliases into DIFFKV_* env; hold weights FP16.
    Also clears the compress-lever keys so each call starts from a known state
    (config['extra_env'] is applied AFTER this, per-call, in run_worker_task)."""
    for k in ("DIFFKV_LAYER_ADAPTIVE_RANK", "DIFFKV_STREAMING_COMPRESS",
              "DIFFKV_COMPRESS_GRAM_SVD", "DIFFKV_RSVD_MAX_RPROJ",
              "DIFFKV_RSVD_OVERSAMPLES", "DIFFKV_RANK_BOOST"):
        os.environ.pop(k, None)
    base = preset
    if preset in ADAPTIVE_PRESETS:
        base, extra = ADAPTIVE_PRESETS[preset]
        for k, v in extra.items():
            os.environ[k] = v
    os.environ["DIFFKV_PRESET"] = base
    # Keep weights FP16 across ALL presets so memory/VRAM comparisons isolate the
    # KV cache, not weight quantization (the low preset otherwise auto-enables
    # 4-bit NF4 weights). Quantization is composable and reported as future work.
    os.environ["DIFFKV_QUANTIZATION"] = "fp16"


def _spawn_and_collect(cmd: List[str], out_path: str, timeout: float) -> Dict[str, Any]:
    """Run `cmd` (a worker that writes its result ATOMICALLY to out_path) and
    return the parsed JSON. Robust to two DiffKV quirks:
      * console spam — the child's stdout/stderr go to DEVNULL (results come from
        the file), so N sequential worker loads don't look like an infinite loop.
        Set DIFFKV_WORKER_VERBOSE=1 to see child logs.
      * hang-at-exit — the DiffKV binary has a known intermittent hang AFTER the
        work + result are done, so we poll for the result file and KILL the child
        once it appears instead of waiting for a clean exit.
    The child inherits _DIFFKV_IN_WORKER=1 so it runs in-process (no re-spawn)."""
    import time as _t
    env = os.environ.copy()
    env["_DIFFKV_IN_WORKER"] = "1"
    quiet = os.environ.get("DIFFKV_WORKER_VERBOSE", "0") != "1"
    stream = subprocess.DEVNULL if quiet else None
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass
    p = subprocess.Popen(cmd, env=env, stdout=stream, stderr=stream)
    deadline = _t.monotonic() + timeout
    try:
        while _t.monotonic() < deadline:
            if os.path.exists(out_path):
                _t.sleep(0.2)          # let the atomic rename settle
                if p.poll() is None:
                    p.kill()           # reap now; do not wait for a hang-at-exit
                break
            if p.poll() is not None:
                break                  # child exited (possibly crashed) on its own
            _t.sleep(0.5)
        else:
            p.kill()
            return {"status": "error", "error": f"worker timed out after {timeout:.0f}s"}
    finally:
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception as e:
            return {"status": "error", "error": f"worker result unreadable: {e}"}
    return {"status": "error", "error": "worker produced no result file (crashed before writing)"}


def _run_worker_isolated(task_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one worker task in a FRESH SUBPROCESS so all GPU memory is released on
    exit. The DiffKV wrapper does not fully free on del/close, so reloading per
    config in-process OOMs a 40GB card (observed: 24.9GB free -> 10.1GB -> OOM).
    Opt in with DIFFKV_ISOLATE_WORKERS=1."""
    import tempfile
    fd, out_path = tempfile.mkstemp(prefix="diffkv_worker_", suffix=".json")
    os.close(fd)
    os.remove(out_path)               # child creates it atomically; must not pre-exist
    cmd = [sys.executable, os.path.abspath(__file__),
           "--worker-task", task_type, "--worker-config", json.dumps(config),
           "--worker-out", out_path]
    timeout = float(os.environ.get("DIFFKV_WORKER_TIMEOUT_S", "1800"))
    try:
        return _spawn_and_collect(cmd, out_path, timeout)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


def run_worker_task(task_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    # Process isolation (opt-in) — run in a subprocess so the GPU is fully freed
    # between configs on memory-tight cards (e.g. 40GB A100). The child sets
    # _DIFFKV_IN_WORKER=1 so it runs the body in-process rather than re-spawning.
    if (os.environ.get("DIFFKV_ISOLATE_WORKERS") == "1"
            and os.environ.get("_DIFFKV_IN_WORKER") != "1"):
        return _run_worker_isolated(task_type, config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
    preset = config.get("preset", "mid")
    gen_len = config.get("gen_len", 128)
    rank = config.get("rank", 32)
    block_size = config.get("block_size", 256)
    n_trials = config.get("n_trials", int(os.environ.get("DIFFKV_BENCH_TRIALS", "3")))

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        full_prompt, ground_truth, answers, match_mode = _build_task(task_type, config, tokenizer)
        ids = tokenizer.encode(full_prompt)
        actual_len = len(ids)
        stop_ids = _derive_stop_ids(tokenizer)

        trials: List[Dict[str, Any]] = []
        last_text = ""

        method_params = config.get("method_params", {})
        if preset in DENSE_FAMILY_METHODS:
            from transformers import AutoModelForCausalLM
            load_kwargs = dict(torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                               device_map="auto" if device == "cuda" else None, trust_remote_code=True)
            # SnapKV needs real attention weights → eager attention path.
            if preset == "snapkv":
                load_kwargs["attn_implementation"] = "eager"
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.eval()
            stop_ids |= _derive_stop_ids(tokenizer)
            CH = int(os.environ.get("DIFFKV_PREFILL_CHUNK_SIZE", "1024"))
            if torch.cuda.is_available():  # warm-up
                with torch.no_grad():
                    _ = model(torch.zeros((1, 8), dtype=torch.long, device=device))
                torch.cuda.synchronize()
            for _ in range(n_trials):
                if preset == "snapkv":
                    tr = _snapkv_trial(model, tokenizer, ids, device, gen_len, stop_ids, CH, **method_params)
                else:
                    tr = _dense_family_trial(model, tokenizer, ids, preset, device, gen_len, stop_ids, CH, method_params)
                trials.append(tr)
                last_text = tr["output_text"]
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            _set_diffkv_env(preset)
            # Per-call DIFFKV_* overrides (e.g. the Gram-eigh compress recipe,
            # used by colab/gram_eigh_decision.py). Applied after the clean slate.
            for _k, _v in config.get("extra_env", {}).items():
                os.environ[_k] = str(_v)
            from serving.hf_diffkv_wrapper import PyTorchDiffKVHFWrapper
            diffkv_config = {"preset": ADAPTIVE_PRESETS.get(preset, (preset,))[0],
                             "rank": rank, "block_size": block_size, "micro_block_size": block_size,
                             "quantization": "fp16"}
            w = PyTorchDiffKVHFWrapper(model_id=model_id, config=diffkv_config,
                                       torch_dtype=torch.float16, device=device)
            w.ensure_loaded()
            stop_ids |= getattr(w, "stop_token_ids", set())
            if torch.cuda.is_available():  # warm-up
                with torch.no_grad():
                    _ = w.model(torch.zeros((1, 8), dtype=torch.long, device=device))
                torch.cuda.synchronize()
            for _ in range(n_trials):
                tr = _diffkv_trial(w, ids, device, gen_len, stop_ids)
                trials.append(tr)
                last_text = tr["output_text"]
            try:
                w.close()
            except Exception:
                pass
            del w
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Aggregate across trials ──
        def agg(key):
            return compute_stats([t[key] for t in trials])
        fwd, comp = agg("prefill_forward_s"), agg("prefill_compress_s")
        dec, tps = agg("decode_time_s"), agg("decode_tps")
        pf_vram = max(t["peak_prefill_vram_gb"] for t in trials)
        dec_vram = max(t["peak_decode_vram_gb"] for t in trials)
        kv_phys = trials[-1]["kv_physical_gb"]
        kv_dense = trials[-1]["kv_dense_equiv_gb"]

        # ── Quality / recall ──
        recall = answer_set_recall(last_text, answers) if answers else 0.0
        passed = (recall >= 0.999) if answers else False
        em = exact_match_score(last_text, ground_truth) if ground_truth else 0.0
        f1 = token_f1_score(last_text, ground_truth) if ground_truth else 0.0

        return {
            "status": "success", "preset": preset, "prompt_len": actual_len, "gen_len": trials[-1]["gen_len"],
            "n_trials": n_trials,
            "prefill_forward_s": fwd["mean"], "prefill_forward_s_ci95": fwd["ci95_margin"],
            "prefill_compress_s": comp["mean"], "prefill_compress_s_ci95": comp["ci95_margin"],
            "prefill_time_s": fwd["mean"] + comp["mean"],
            "decode_time_s": dec["mean"], "decode_time_s_ci95": dec["ci95_margin"],
            "decode_tps": tps["mean"], "decode_tps_ci95": tps["ci95_margin"],
            "peak_prefill_vram_gb": pf_vram, "peak_decode_vram_gb": dec_vram,
            "kv_physical_gb": kv_phys, "kv_dense_equiv_gb": kv_dense,
            "kv_footprint_realized": preset not in ("int8_kv", "kivi2"),
            "compression_ratio": (kv_dense / kv_phys) if kv_phys > 0 else 1.0,
            "output_text": last_text, "ground_truth": ground_truth,
            "answers": answers, "match_mode": match_mode,
            "recall": recall * 100.0, "passed": passed,
            "exact_match": em, "f1_score": f1,
        }
    except Exception as e:
        err = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[WORKER_ERROR] {err}", file=sys.stderr)
        return {"status": "error", "error": err}


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODS = ["dense", "int8_kv", "kivi2", "streaming", "keynorm_hh", "snapkv", "low", "mid", "high"]


def exp1_memory_vs_context(model_id: str, contexts: List[int]) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 1: Memory vs Context (all methods, consistent metric)\n" + "=" * 80)
    results = {}
    for ctx in contexts:
        results[ctx] = {}
        for m in ALL_METHODS:
            print(f"   -> ctx={ctx}, method={m} ...")
            res = run_worker_task("mem_ctx", {"model_id": model_id, "preset": m, "ctx_len": ctx, "gen_len": 32})
            results[ctx][m] = res
            if res.get("status") == "success":
                print(f"      peak_prefill={res['peak_prefill_vram_gb']:.2f}GB | "
                      f"KV_phys={res['kv_physical_gb']:.3f}GB | ratio={res['compression_ratio']:.2f}x")
    return results


def exp2_throughput_vs_context(model_id: str, contexts: List[int]) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 2: Throughput vs Context (dense vs DiffKV-mid, matched decode loop)\n" + "=" * 80)
    results = {}
    for ctx in contexts:
        rd = run_worker_task("tps_ctx", {"model_id": model_id, "preset": "dense", "ctx_len": ctx, "gen_len": 64})
        rk = run_worker_task("tps_ctx", {"model_id": model_id, "preset": "mid", "ctx_len": ctx, "gen_len": 64})
        results[ctx] = {"dense": rd, "diffkv_mid": rk}
        if rk.get("status") == "success" and rd.get("status") == "success":
            print(f"   ctx={ctx}: dense {rd['decode_tps']:.1f}±{rd['decode_tps_ci95']:.1f} tps | "
                  f"DiffKV {rk['decode_tps']:.1f}±{rk['decode_tps_ci95']:.1f} tps | "
                  f"prefill fwd {rk['prefill_forward_s']:.2f}s + comp {rk['prefill_compress_s']:.2f}s")
    return results


def exp3_long_context_quality(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 3: Long-Context Quality (context-grounded, multi-metric)\n" + "=" * 80)
    eval_prompts = [
        {"doc": "NAT Paper", "context": load_file_text(NAT_PAPER_PATH),
         "q": "What is the primary difference between Neighborhood Attention and Window Self-Attention?",
         "concepts": ["overlapping|overlapping windows|overlapping neighborhoods"]},
        {"doc": "Berry Paper", "context": load_file_text(BERRY_PAPER_PATH),
         "q": "What mathematical bound is analyzed in the text?",
         "concepts": ["berry-esseen|berry esseen bound|normal approximation"]},
        {"doc": "Random Features Paper", "context": load_file_text(RANDOM_PAPER_PATH),
         "q": "What mapping technique is introduced to approximate kernel functions?",
         "concepts": ["random fourier|fourier features|random features"]},
    ]
    results = {}
    for item in eval_prompts:
        results[item["doc"]] = {}
        for m in ALL_METHODS:
            res = run_worker_task("quality", {"model_id": model_id, "preset": m, "ctx_len": 8192, "gen_len": 128,
                                              "context_text": item["context"], "question": item["q"],
                                              "ground_truth": item["concepts"][0].split("|")[0]})
            res["concept_recall_pct"] = concept_synonym_recall(res.get("output_text", ""), item["concepts"])
            results[item["doc"]][m] = res
            print(f"   {item['doc']:<22} {m:<11} EM={res.get('exact_match',0):.0f}% "
                  f"concept={res['concept_recall_pct']:.0f}%")
    return results


def exp4_end_to_end_tradeoff(model_id: str, ctx: int = 16384) -> Dict[str, Any]:
    """Accuracy-vs-memory tradeoff at ONE matched context length (fixes the 15.0
    bug that paired 16K memory with 8K quality)."""
    print("\n" + "=" * 80 + f"\n🔥 EXP 4: Quality/Memory Tradeoff @ {ctx} (matched context)\n" + "=" * 80)
    tradeoff = {}
    for m in ALL_METHODS:
        # Memory from a NIAH run at this ctx; quality = recall on the same run.
        res = run_worker_task("niah", {"model_id": model_id, "preset": m, "ctx_len": ctx, "depth": 0.5,
                                       "needle_code": "APEX-3391-SHIELD", "gen_len": 48})
        tradeoff[m] = {"kv_physical_gb": res.get("kv_physical_gb", 0.0),
                       "peak_prefill_vram_gb": res.get("peak_prefill_vram_gb", 0.0),
                       "compression_ratio": res.get("compression_ratio", 0.0),
                       "recall_pct": res.get("recall", 0.0)}
        print(f"   {m:<11} KV_phys={tradeoff[m]['kv_physical_gb']:.3f}GB "
              f"peakVRAM={tradeoff[m]['peak_prefill_vram_gb']:.2f}GB recall={tradeoff[m]['recall_pct']:.0f}%")
    return tradeoff


def exp5_needle_in_haystack(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 5: NIAH grid — dense vs DiffKV, multi-needle + Wilson CI\n" + "=" * 80)
    depths = [0.10, 0.30, 0.50, 0.70, 0.90]
    contexts = [4096, 8192, 16384, 32768, 65536]
    samples = int(os.environ.get("DIFFKV_NIAH_SAMPLES", "3"))
    needles = generate_random_needles(len(depths) * len(contexts) * samples * 2 + 10)
    results = {}
    for method in ("dense", "mid"):
        results[method] = {}
        succ = tot = 0
        idx = 0
        for ctx in contexts:
            results[method][ctx] = {}
            for d in depths:
                cell_pass = 0
                for _ in range(samples):
                    code, sent = needles[idx % len(needles)]
                    idx += 1
                    res = run_worker_task("niah", {"model_id": model_id, "preset": method, "ctx_len": ctx,
                                                   "depth": d, "needle_code": code, "needle_sent": sent, "gen_len": 32})
                    ok = code.upper() in res.get("output_text", "").upper()
                    cell_pass += int(ok)
                    succ += int(ok)
                    tot += 1
                ci = wilson_ci(cell_pass, samples)
                results[method][ctx][str(d)] = {"pass": cell_pass, "n": samples, "recall_pct": ci["p"]}
                print(f"   [{method}] ctx={ctx:<6} depth={int(d*100)}% : {cell_pass}/{samples}")
        overall = wilson_ci(succ, tot)
        results[method]["overall"] = overall
        print(f"   -> {method}: {succ}/{tot} = {overall['p']:.1f}% "
              f"(95% CI [{overall['low']:.1f}, {overall['high']:.1f}])")
    return results


def exp5b_ruler(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 5b: RULER-style suite (dense vs DiffKV-mid) + Wilson CI\n" + "=" * 80)
    tasks = ["niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt", "fwe"]
    contexts = [8192, 16384, 32768]
    samples = int(os.environ.get("DIFFKV_RULER_SAMPLES", "5"))
    results = {}
    for method in ("dense", "mid"):
        results[method] = {}
        for task in tasks:
            results[method][task] = {}
            for ctx in contexts:
                recalls, passes = [], 0
                for s in range(samples):
                    res = run_worker_task("ruler", {"model_id": model_id, "preset": method, "ctx_len": ctx,
                                                    "ruler_task": task, "seed": 1000 * ctx + s, "gen_len": 64})
                    recalls.append(res.get("recall", 0.0))
                    passes += int(res.get("passed", False))
                ci = wilson_ci(passes, samples)
                mr = sum(recalls) / len(recalls) if recalls else 0.0
                results[method][task][ctx] = {"pass_rate_pct": ci["p"], "pass_ci95": ci["margin"],
                                              "mean_recall_pct": mr, "n": samples}
                print(f"   [{method}] {task:<16} ctx={ctx:<6} pass={ci['p']:.0f}%±{ci['margin']:.0f} "
                      f"recall={mr:.0f}%")
    return results


def exp6_model_scale_comparison(model_a: str, model_b: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 6: Model / family generalization\n" + "=" * 80)
    results = {}
    for m in [model_a, model_b]:
        r = {}
        for method in ("dense", "mid"):
            res = run_worker_task("niah", {"model_id": m, "preset": method, "ctx_len": 16384, "depth": 0.5,
                                           "needle_code": "TITAN-5567-ORBIT", "gen_len": 48})
            r[method] = res
            print(f"   {m} [{method}] peakVRAM={res.get('peak_prefill_vram_gb',0):.2f}GB "
                  f"KV_phys={res.get('kv_physical_gb',0):.3f}GB tps={res.get('decode_tps',0):.1f} "
                  f"recall={res.get('recall',0):.0f}%")
        results[m] = r
    return results


def exp7_ablation_study(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 7: Ablation (presets + adaptive variants)\n" + "=" * 80)
    results = {}
    for abl in ["low", "mid", "high", "adaptive_rank", "adaptive_stream"]:
        res = run_worker_task("niah", {"model_id": model_id, "preset": abl, "ctx_len": 16384, "depth": 0.5,
                                       "needle_code": "CYPHER-9102-PRIME", "gen_len": 48})
        results[abl] = res
        print(f"   {abl:<16} peakVRAM={res.get('peak_prefill_vram_gb',0):.2f}GB "
              f"KV_phys={res.get('kv_physical_gb',0):.3f}GB tps={res.get('decode_tps',0):.1f} "
              f"recall={res.get('recall',0):.0f}%")
    return results


def exp8_decode_length_scaling(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 8: Decode-length scaling (dense vs DiffKV-mid)\n" + "=" * 80)
    results = {}
    for gen_l in [128, 256, 512, 1024]:
        results[gen_l] = {}
        for method in ("dense", "mid"):
            res = run_worker_task("gen_scale", {"model_id": model_id, "preset": method, "ctx_len": 8192, "gen_len": gen_l})
            results[gen_l][method] = res
            print(f"   gen={gen_l:<5} [{method}] tps={res.get('decode_tps',0):.1f}±{res.get('decode_tps_ci95',0):.1f} "
                  f"peak_decode={res.get('peak_decode_vram_gb',0):.2f}GB")
    return results


def exp9_nsight_systems_profiling() -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 9: Nsight Systems timeline (best-effort)\n" + "=" * 80)
    trace = os.path.join(REPO, "diffkv_nsys_trace")
    cmd = ["nsys", "profile", "-t", "cuda,nvtx,osrt", "-s", "cpu", "--stats=true", "--force-overwrite=true",
           "-o", trace, sys.executable, os.path.join(HERE, "run_nat_eval.py"),
           "--worker", "mid_preset", "--model", "Qwen/Qwen2.5-7B-Instruct"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # nsys --stats prints tables like "Time (%)  Total Time (ns)  ... Name".
        # Grab the CUDA-kernel and memops summary rows rather than a bespoke string.
        kernel_rows = [ln.strip() for ln in p.stdout.splitlines()
                       if re.search(r"\d+\.\d+\s+\d+\s+\d+", ln)][:15]
        return {"status": "success", "command": " ".join(cmd),
                "summary_rows": kernel_rows, "stdout_tail": p.stdout[-800:]}
    except Exception as e:
        print(f"   -> nsys unavailable: {e}")
        return {"status": "cli_not_found", "command": " ".join(cmd), "error": str(e)}


def exp10_nsight_compute_profiling() -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 10: Nsight Compute occupancy (best-effort)\n" + "=" * 80)
    out_csv = os.path.join(REPO, "diffkv_ncu_report.csv")
    cmd = ["ncu", "--csv", "--log-file", out_csv, "--target-processes", "all",
           "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
                        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
                        "sm__warps_active.avg.pct_of_peak_sustained_active",
           sys.executable, os.path.join(HERE, "profile_decode_step.py")]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        metrics = {}
        for ln in p.stdout.splitlines():
            if "sm__throughput" in ln:
                m = re.search(r"([\d\.]+)\s*$", ln)
                if m:
                    metrics["sm_throughput_pct"] = float(m.group(1))
            if "dram_throughput" in ln:
                m = re.search(r"([\d\.]+)\s*$", ln)
                if m:
                    metrics["dram_throughput_pct"] = float(m.group(1))
        return {"status": "success", "command": " ".join(cmd), "structured_metrics": metrics,
                "stdout_tail": p.stdout[-800:]}
    except Exception as e:
        print(f"   -> ncu unavailable: {e}")
        return {"status": "cli_not_found", "command": " ".join(cmd), "error": str(e)}


def exp11_cuda_memory_allocation() -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 11: CUDA allocator stats\n" + "=" * 80)
    if not torch.cuda.is_available():
        return {"status": "cuda_not_available"}
    ms = torch.cuda.memory_stats()
    alloc = torch.cuda.memory_allocated() / 1e9
    resv = torch.cuda.memory_reserved() / 1e9
    frag = (1.0 - (alloc / resv)) * 100.0 if resv > 0 else 0.0
    stats = {"allocated_gb": alloc, "reserved_gb": resv, "fragmentation_pct": frag,
             "num_alloc_retries": ms.get("num_alloc_retries", 0),
             "cuda_malloc_count": ms.get("allocation.all.allocated", 0)}
    print(f"   alloc={alloc:.2f}GB reserved={resv:.2f}GB frag={frag:.1f}%")
    return stats


def exp12_longbench_evaluation(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 12: Context-grounded QA (synthetic; plug real LongBench here)\n" + "=" * 80)
    # HONEST NAMING: these are ad-hoc context-grounded QA probes over our 3 real
    # papers, NOT the LongBench dataset. To run real LongBench, load THU-KEG/
    # LongBench via `datasets` and feed each sample's context/question/answers
    # through run_worker_task(task_type="quality", ...). Left as a documented hook
    # so the paper does not misrepresent ad-hoc prompts as LongBench.
    tasks = [
        {"name": "single_doc_qa", "q": "What is the primary technical claim of the text?",
         "gt": "Neighborhood Attention", "context": load_file_text(NAT_PAPER_PATH)},
        {"name": "math_bound_qa", "q": "What bound does the text analyze?",
         "gt": "Berry-Esseen", "context": load_file_text(BERRY_PAPER_PATH)},
        {"name": "method_qa", "q": "What random-feature technique is analyzed?",
         "gt": "random Fourier features", "context": load_file_text(RANDOM_PAPER_PATH)},
        {"name": "injected_kv_retrieval", "q": "What value is mapped to key ID-9923?",
         "gt": "SIGMA-9923-BETA", "context": "KV Store:\nID-9923: SIGMA-9923-BETA\nID-1105: THETA-1105-ALPHA\n"
         + load_file_text(NAT_PAPER_PATH)},
    ]
    results = {}
    for t in tasks:
        res = run_worker_task("quality", {"model_id": model_id, "preset": "mid", "ctx_len": 16384, "gen_len": 64,
                                          "context_text": t["context"], "question": t["q"], "ground_truth": t["gt"]})
        results[t["name"]] = {"exact_match": res.get("exact_match", 0.0), "f1_score": res.get("f1_score", 0.0)}
        print(f"   {t['name']:<20} EM={res.get('exact_match',0):.0f}% F1={res.get('f1_score',0):.0f}%")
    return results


def exp13_different_documents(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 13: Per-corpus retrieval (honest corpus names)\n" + "=" * 80)
    # Honest: these are the three real technical corpora we ship — no fake
    # "legal / 10-K / fiction" relabeling of the same papers.
    corpora = [
        {"name": "NAT (attention paper)", "context": load_file_text(NAT_PAPER_PATH), "gt": "Neighborhood Attention"},
        {"name": "Berry-Esseen (stats)", "context": load_file_text(BERRY_PAPER_PATH), "gt": "Berry-Esseen"},
        {"name": "Random Features (kernels)", "context": load_file_text(RANDOM_PAPER_PATH), "gt": "random Fourier features"},
    ]
    results = {}
    for c in corpora:
        res = run_worker_task("doc_domain", {"model_id": model_id, "preset": "mid", "ctx_len": 8192, "gen_len": 64,
                                             "context_text": c["context"],
                                             "question": f"State the central concept of this text.",
                                             "ground_truth": c["gt"]})
        results[c["name"]] = {"exact_match": res.get("exact_match", 0.0)}
        print(f"   {c['name']:<28} EM={res.get('exact_match',0):.0f}%")
    return results


def exp14_prompt_categories(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 14: 20-probe category suite (retrieval uses injected facts)\n" + "=" * 80)
    nat, berry, rand = load_file_text(NAT_PAPER_PATH), load_file_text(BERRY_PAPER_PATH), load_file_text(RANDOM_PAPER_PATH)
    probes = [
        {"id": 1, "cat": "Retrieval", "q": "What is the passcode?", "gt": "VORTEX-4412-MATRIX", "ctx": "Passcode: VORTEX-4412-MATRIX.\n" + nat},
        {"id": 2, "cat": "Retrieval", "q": "What is the clause ID?", "gt": "CLAUSE-8891-SEC", "ctx": "Clause ID: CLAUSE-8891-SEC.\n" + berry},
        {"id": 3, "cat": "Retrieval", "q": "What is the ticket number?", "gt": "TICKET-4402-Z", "ctx": "Ticket: TICKET-4402-Z.\n" + rand},
        {"id": 4, "cat": "Retrieval", "q": "What is the hash key?", "gt": "HASH-9912-KEY", "ctx": "Hash Key: HASH-9912-KEY.\n" + nat},
        {"id": 5, "cat": "Comparison", "q": "Compare overlapping vs shifted windows.", "gt": "overlapping", "ctx": nat},
        {"id": 6, "cat": "Comparison", "q": "Does the bound concern sums or products?", "gt": "sums", "ctx": berry},
        {"id": 7, "cat": "Comparison", "q": "Is the feature map linear or nonlinear?", "gt": "nonlinear", "ctx": rand},
        {"id": 8, "cat": "Comparison", "q": "Compare linear vs quadratic complexity.", "gt": "linear", "ctx": nat},
        {"id": 9, "cat": "Summary", "q": "Summarize the core thesis.", "gt": "Neighborhood Attention", "ctx": nat},
        {"id": 10, "cat": "Summary", "q": "Summarize the mathematical bound.", "gt": "Berry-Esseen", "ctx": berry},
        {"id": 11, "cat": "Summary", "q": "Summarize the kernel approximation method.", "gt": "random Fourier features", "ctx": rand},
        {"id": 12, "cat": "Summary", "q": "What is the injected reduction code?", "gt": "REDUCE-7781", "ctx": "Reduction code: REDUCE-7781.\n" + nat},
        {"id": 13, "cat": "Reasoning", "q": "What happens to memory as the window grows?", "gt": "increases", "ctx": nat},
        {"id": 14, "cat": "Reasoning", "q": "As sample size grows, the approximation error?", "gt": "decreases", "ctx": berry},
        {"id": 15, "cat": "Reasoning", "q": "Do more random features improve the approximation?", "gt": "yes", "ctx": rand},
        {"id": 16, "cat": "Reasoning", "q": "What term names the reference token in delta coding?", "gt": "anchor", "ctx": "In delta coding the reference token is the anchor.\n" + nat},
        {"id": 17, "cat": "Extraction", "q": "Extract the secret passcode.", "gt": "NEXUS-8812-PRIME", "ctx": "Code: NEXUS-8812-PRIME.\n" + berry},
        {"id": 18, "cat": "Extraction", "q": "Extract the project codename.", "gt": "CODENAME-AURORA", "ctx": "Codename: CODENAME-AURORA.\n" + nat},
        {"id": 19, "cat": "Extraction", "q": "Extract the transform name.", "gt": "Fourier", "ctx": rand},
        {"id": 20, "cat": "Extraction", "q": "Extract the injected device ID.", "gt": "DEV-5590-X", "ctx": "Device: DEV-5590-X.\n" + nat},
    ]
    results = {}
    for p in probes:
        res = run_worker_task("category", {"model_id": model_id, "preset": "mid", "ctx_len": 8192, "gen_len": 64,
                                           "context_text": p["ctx"], "question": p["q"], "ground_truth": p["gt"]})
        results[f"probe_{p['id']}_{p['cat']}"] = {"probe_id": p["id"], "category": p["cat"],
                                                  "exact_match": res.get("exact_match", 0.0)}
        print(f"   #{p['id']:<2} {p['cat']:<11} EM={res.get('exact_match',0):.0f}%")
    return results


def exp15_batch_size_scaling(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 15: Isolated batched forward-pass scaling\n" + "=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device, trust_remote_code=True).eval()
        for bs in [1, 2, 4, 8]:
            prompts = [f"Summarize paper contribution #{i+1}." for i in range(bs)]
            inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs, use_cache=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            ntok = inputs.input_ids.numel()
            results[bs] = {"forward_latency_s": dt, "forward_throughput_tps": ntok / dt if dt > 0 else 0.0,
                           "peak_vram_gb": peak}
            print(f"   BS={bs:<2} latency={dt:.4f}s tps={ntok/dt if dt>0 else 0:.0f} peak={peak:.2f}GB")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results
    except Exception as e:
        return {"status": "error", "error": str(e)}


def exp16_long_conversations(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 16: Multi-turn growth (DiffKV-mid)\n" + "=" * 80)
    turns = 10
    metrics = []
    for t in range(1, turns + 1):
        res = run_worker_task("mem_ctx", {"model_id": model_id, "preset": "mid", "ctx_len": 1000 * t, "gen_len": 32})
        lat = res.get("prefill_time_s", 0) + res.get("decode_time_s", 0)
        metrics.append({"turn": t, "ctx_len": res.get("prompt_len", 1000 * t), "latency_s": lat,
                        "kv_physical_gb": res.get("kv_physical_gb", 0), "vram_gb": res.get("peak_prefill_vram_gb", 0)})
        print(f"   turn {t:<2} len={metrics[-1]['ctx_len']:<6} lat={lat:.2f}s "
              f"KV_phys={metrics[-1]['kv_physical_gb']:.3f}GB VRAM={metrics[-1]['vram_gb']:.2f}GB")
    return {"total_turns": turns, "turn_metrics": metrics}


def exp17_multi_document_qa(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 17: Multi-document cross-referencing (injected code)\n" + "=" * 80)
    ctx = ("--- DOC 1: NAT ---\n" + load_file_text(NAT_PAPER_PATH) + "\n\n"
           "--- DOC 2: Berry ---\n" + load_file_text(BERRY_PAPER_PATH) + "\n\n"
           "--- DOC 3: Random Features ---\nThe cross-reference security code is ALPHA-9981-VECTOR.\n"
           + load_file_text(RANDOM_PAPER_PATH))
    res = run_worker_task("quality", {"model_id": model_id, "preset": "mid", "ctx_len": 16384, "gen_len": 48,
                                      "context_text": ctx, "question": "State the cross-reference security code exactly.",
                                      "ground_truth": "ALPHA-9981-VECTOR"})
    print(f"   Multi-doc EM={res.get('exact_match',0):.0f}%")
    return {"exact_match": res.get("exact_match", 0.0), "metrics": res}


def exp18_reconstruction_fidelity(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 18: Per-layer compressor fidelity (all layers)\n" + "=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device, trust_remote_code=True).eval()
        inputs = tok(load_file_text(NAT_PAPER_PATH)[:2000], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        legacy = _cache_to_legacy(out.past_key_values)
        if not legacy:
            return {"status": "error", "error": "past_key_values empty"}
        from native_core.compression.lowrank import compress_lowrank
        layer_results = []
        for l_idx, (K, V) in enumerate(legacy):
            K_layer = K[0, 0].float()
            anchor = K_layer[0:1, :]
            delta = K_layer - anchor
            lr = compress_lowrank(delta, rank=32)
            recon = anchor + (lr.U.float() @ lr.V.float()) * lr.scale
            cos = torch.nn.functional.cosine_similarity(K_layer.flatten(), recon.flatten(), dim=0).item()
            mse = torch.mean((K_layer - recon) ** 2).item()
            rel = (torch.norm(K_layer - recon) / torch.norm(K_layer)).item() * 100.0
            layer_results.append({"layer": l_idx, "cosine_similarity": cos, "mse_loss": mse, "relative_error_pct": rel})
        cos_s = compute_stats([r["cosine_similarity"] for r in layer_results])
        rel_s = compute_stats([r["relative_error_pct"] for r in layer_results])
        print(f"   layers={len(layer_results)} cos={cos_s['mean']:.5f}±{cos_s['std']:.5f} "
              f"rel_err={rel_s['mean']:.2f}%±{rel_s['std']:.2f}%")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"num_layers_evaluated": len(layer_results),
                "mean_cosine_similarity": cos_s["mean"], "std_cosine_similarity": cos_s["std"],
                "mean_relative_error_pct": rel_s["mean"], "std_relative_error_pct": rel_s["std"],
                "layer_breakdown": layer_results}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def exp19_sensitivity_analysis(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 19: Rank × BlockSize × Preset × Context sweep\n" + "=" * 80)
    results = {}
    for r in [16, 32, 64]:
        for b in [128, 256]:
            for p in ["low", "mid", "high"]:
                for c in [4096, 8192]:
                    key = f"r{r}_b{b}_{p}_c{c}"
                    res = run_worker_task("mem_ctx", {"model_id": model_id, "preset": p, "ctx_len": c,
                                                      "rank": r, "block_size": b, "gen_len": 32})
                    results[key] = {"rank": r, "block_size": b, "preset": p, "context_len": c,
                                    "kv_physical_gb": res.get("kv_physical_gb", 0.0),
                                    "compression_ratio": res.get("compression_ratio", 0.0),
                                    "tps": res.get("decode_tps", 0.0),
                                    "recall_or_prompt_len": res.get("prompt_len", 0)}
    print(f"   completed {len(results)} grid points "
          f"(KV_phys should INCREASE with rank — verifies the rank axis is live)")
    return results


def exp20_system_stability(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 20: Stability stress (10 runs)\n" + "=" * 80)
    runs, passed, failures = 10, 0, []
    for i in range(1, runs + 1):
        res = run_worker_task("mem_ctx", {"model_id": model_id, "preset": "mid", "ctx_len": 8192, "gen_len": 64})
        if res.get("status") == "success" and len(res.get("output_text", "")) > 0:
            passed += 1
        else:
            failures.append(f"run {i}: {res.get('error', 'empty output')[:120]}")
    ci = wilson_ci(passed, runs)
    print(f"   stability {passed}/{runs} = {ci['p']:.0f}% (95% CI [{ci['low']:.0f}, {ci['high']:.0f}])")
    return {"total_runs": runs, "passed_runs": passed, "failures": failures,
            "success_rate_pct": ci["p"], "success_ci95": ci["margin"]}


def exp21_external_baselines_comparison(model_id: str) -> Dict[str, Any]:
    print("\n" + "=" * 80 + "\n🔥 EXP 21: All baselines head-to-head @16K NIAH (consistent metric)\n" + "=" * 80)
    results = {}
    for base in ALL_METHODS:
        res = run_worker_task("niah", {"model_id": model_id, "preset": base, "ctx_len": 16384, "depth": 0.5,
                                       "needle_code": "VORTEX-7712-PRIME", "gen_len": 48})
        results[base] = res
        if res.get("status") == "success":
            print(f"   {base:<11} KV_phys={res['kv_physical_gb']:.3f}GB ratio={res['compression_ratio']:.2f}x "
                  f"peakVRAM={res['peak_prefill_vram_gb']:.2f}GB recall={res['recall']:.0f}% "
                  f"tps={res['decode_tps']:.1f}±{res['decode_tps_ci95']:.1f}")
        else:
            print(f"   {base:<11} ERROR: {res.get('error','')[:100]}")
    return results


def _avg_recall(model_id: str, method: str, ctx: int, ruler_task_name: str,
                samples: int, method_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one method at one operating point over `samples` seeds; return mean
    recall + KV footprint + realized flag. Shared by the frontier / curve exps."""
    recs, kv, real, tps = [], 0.0, True, 0.0
    for s in range(samples):
        cfg = {"model_id": model_id, "preset": method, "ctx_len": ctx, "ruler_task": ruler_task_name,
               "seed": 7000 * ctx + s, "gen_len": 64}
        if method_params:
            cfg["method_params"] = method_params
        r = run_worker_task("ruler", cfg)
        if r.get("status") != "success":
            continue
        recs.append(r.get("recall", 0.0))
        kv = r.get("kv_physical_gb", 0.0)
        real = r.get("kv_footprint_realized", True)
        tps = r.get("decode_tps", 0.0)
    return {"mean_recall_pct": (sum(recs) / len(recs) if recs else 0.0),
            "kv_physical_gb": kv, "kv_footprint_realized": real, "decode_tps": tps, "n": len(recs)}


def exp22_memory_quality_frontier(model_id: str, ctx: int = 32768) -> Dict[str, Any]:
    """THE make-or-break comparison: sweep each method across its memory dial at a
    fixed long context and plot accuracy vs KV footprint. DiffKV is a real
    contribution ONLY if it occupies a point the strong baselines (KIVI, SnapKV)
    do NOT dominate — i.e. matches dense quality at a memory budget where they
    have already degraded. Uses niah_multikey (the discriminating retrieval task).
    """
    print("\n" + "=" * 80 + f"\n🔥 EXP 22: Accuracy / KV-memory FRONTIER @ {ctx} (niah_multikey)\n" + "=" * 80)
    samples = int(os.environ.get("DIFFKV_FRONTIER_SAMPLES", "3"))
    task = "niah_multikey"
    points = []

    def add(label, method, params=None):
        r = _avg_recall(model_id, method, ctx, task, samples, params)
        points.append({"label": label, "method": method, "params": params or {},
                       "kv_physical_gb": r["kv_physical_gb"], "recall_pct": r["mean_recall_pct"],
                       "kv_footprint_realized": r["kv_footprint_realized"], "decode_tps": r["decode_tps"]})
        print(f"   {label:<16} KV={r['kv_physical_gb']:.3f}GB recall={r['mean_recall_pct']:.0f}% "
              f"({'realized' if r['kv_footprint_realized'] else 'theoretical'})")

    add("dense", "dense")                                             # reference (max mem, target quality)
    for p in ["high", "mid", "low"]:
        add(f"DiffKV-{p}", p)
    for bits in [4, 3, 2]:
        add(f"KIVI-{bits}bit", "kivi2", {"bits": bits})
    for b in [1024, 512, 256]:
        add(f"SnapKV-{b}", "snapkv", {"budget": b})
    for b in [512, 256, 128]:
        add(f"KeyNormHH-{b}", "keynorm_hh", {"heavy_hitter_budget": b})
    for rec in [1024, 512, 256]:
        add(f"Streaming-{rec}", "streaming", {"recency_window": rec})
    return {"context_len": ctx, "task": task, "samples": samples, "points": points}


def exp23_quality_vs_context(model_id: str) -> Dict[str, Any]:
    """The headline curve: recall vs context length for dense and the strong
    baselines. The claim survives iff DiffKV tracks dense out to long context
    while KIVI/SnapKV/streaming peel away."""
    print("\n" + "=" * 80 + "\n🔥 EXP 23: Quality vs Context (dense vs DiffKV vs strong baselines)\n" + "=" * 80)
    samples = int(os.environ.get("DIFFKV_CURVE_SAMPLES", "3"))
    contexts = [8192, 16384, 32768, 65536]
    methods = [("dense", "dense", None), ("DiffKV-mid", "mid", None),
               ("KIVI-2bit", "kivi2", {"bits": 2}), ("SnapKV-512", "snapkv", {"budget": 512}),
               ("Streaming-512", "streaming", {"recency_window": 512})]
    task = "niah_multikey"
    results = {}
    for label, method, params in methods:
        results[label] = {}
        for ctx in contexts:
            r = _avg_recall(model_id, method, ctx, task, samples, params)
            results[label][ctx] = {"recall_pct": r["mean_recall_pct"], "kv_physical_gb": r["kv_physical_gb"]}
            print(f"   {label:<15} ctx={ctx:<6} recall={r['mean_recall_pct']:.0f}% KV={r['kv_physical_gb']:.3f}GB")
    return {"task": task, "contexts": contexts, "samples": samples, "curves": results}


def build_pareto(exp21: Dict[str, Any]) -> Dict[str, Any]:
    """Accuracy-vs-memory Pareto points (the headline figure), all methods on the
    SAME axes and the SAME measurement."""
    pareto = {}
    for m, r in exp21.items():
        if isinstance(r, dict) and r.get("status") == "success":
            pareto[m] = {"kv_physical_gb": r["kv_physical_gb"], "peak_prefill_vram_gb": r["peak_prefill_vram_gb"],
                         "recall_pct": r["recall"], "decode_tps": r["decode_tps"],
                         "compression_ratio": r["compression_ratio"]}
    return pareto


# ─────────────────────────────────────────────────────────────────────────────
# Master
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DiffKV research paper benchmark runner (REWRITE 16.0)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-14b", dest="model_b", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Second model FAMILY for generalization (default: Llama-3.1-8B, not just a bigger Qwen).")
    parser.add_argument("--out", default="diffkv_paper_benchmark_results.json")
    parser.add_argument("--only", default="", help="comma-separated experiment ids to run (e.g. 1,5,21). Empty = all.")
    parser.add_argument("--worker-task", default="")
    parser.add_argument("--worker-config", default="{}")
    parser.add_argument("--worker-out", default="", help="isolated-worker result file (JSON)")
    args = parser.parse_args()

    if args.worker_task:
        res = run_worker_task(args.worker_task, json.loads(args.worker_config))
        if args.worker_out:
            # Atomic write: the parent polls for this file then kills us (the
            # binary can hang at exit), so it must never observe a partial file.
            tmp = args.worker_out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(res, f)
            os.replace(tmp, args.worker_out)
        else:
            print(json.dumps(res))
        return

    print("=" * 73)
    print("  DIFFERENTIAL-KV (DiffKV) A100 RESEARCH PAPER BENCHMARK — REWRITE 16.0")
    print(f"  Primary: {args.model}  |  Second family: {args.model_b}")
    print("=" * 73)

    contexts = [4096, 8192, 16384, 32768, 65536]
    only = set(x.strip() for x in args.only.split(",") if x.strip())

    def want(i):
        return (not only) or (str(i) in only)

    R = {}
    if want(1): R["exp1_memory_vs_context"] = exp1_memory_vs_context(args.model, contexts)
    if want(2): R["exp2_throughput_vs_context"] = exp2_throughput_vs_context(args.model, contexts)
    if want(3): R["exp3_long_context_quality"] = exp3_long_context_quality(args.model)
    if want(4): R["exp4_end_to_end_tradeoff"] = exp4_end_to_end_tradeoff(args.model, 16384)
    if want(5): R["exp5_needle_in_haystack"] = exp5_needle_in_haystack(args.model)
    if want("5b"): R["exp5b_ruler"] = exp5b_ruler(args.model)
    if want(6): R["exp6_model_scale"] = exp6_model_scale_comparison(args.model, args.model_b)
    if want(7): R["exp7_ablation"] = exp7_ablation_study(args.model)
    if want(8): R["exp8_decode_scaling"] = exp8_decode_length_scaling(args.model)
    if want(9): R["exp9_nsys"] = exp9_nsight_systems_profiling()
    if want(10): R["exp10_ncu"] = exp10_nsight_compute_profiling()
    if want(11): R["exp11_cuda_mem"] = exp11_cuda_memory_allocation()
    if want(12): R["exp12_longbench"] = exp12_longbench_evaluation(args.model)
    if want(13): R["exp13_doc_domains"] = exp13_different_documents(args.model)
    if want(14): R["exp14_prompt_categories"] = exp14_prompt_categories(args.model)
    if want(15): R["exp15_batch_scaling"] = exp15_batch_size_scaling(args.model)
    if want(16): R["exp16_long_conversations"] = exp16_long_conversations(args.model)
    if want(17): R["exp17_multidoc_qa"] = exp17_multi_document_qa(args.model)
    if want(18): R["exp18_reconstruction_fidelity"] = exp18_reconstruction_fidelity(args.model)
    if want(19): R["exp19_sensitivity"] = exp19_sensitivity_analysis(args.model)
    if want(20): R["exp20_stability"] = exp20_system_stability(args.model)
    if want(21): R["exp21_external_baselines"] = exp21_external_baselines_comparison(args.model)
    if want(22): R["exp22_memory_quality_frontier"] = exp22_memory_quality_frontier(args.model, 32768)
    if want(23): R["exp23_quality_vs_context"] = exp23_quality_vs_context(args.model)
    if "exp21_external_baselines" in R:
        R["pareto_accuracy_vs_memory"] = build_pareto(R["exp21_external_baselines"])

    out_file = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(os.getcwd(), args.out))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\n✅ Done. Raw JSON → {out_file}")

    repo_out = os.path.join(REPO, os.path.basename(args.out))
    if os.path.abspath(repo_out) != os.path.abspath(out_file):
        try:
            with open(repo_out, "w") as f:
                json.dump(R, f, indent=2)
            print(f"✅ Backup copy → {repo_out}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
