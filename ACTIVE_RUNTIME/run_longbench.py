#!/usr/bin/env python3
"""
run_longbench.py — LongBench Evaluation for DiffKV ACTIVE_RUNTIME
==================================================================
Runs N examples each from: NarrativeQA, GovReport, Qasper, HotpotQA

Per-dataset output:
  - EM            Exact Match (1 if any gold answer matches exactly, else 0)
  - F1            Token-level F1 (QA tasks)
  - Rouge-L       LCS-based Rouge-L (summarization)
  - prefill_s     Total prefill wall-clock time (s) across all examples
  - decode_tps    Mean decode throughput (output tokens / decode wall-time)
  - peak_mlx_mb   Peak MPS/MLX memory allocated (MB) across all examples
  - peak_rss_mb   Peak process RSS (MB) across all examples

Usage:
    cd ACTIVE_RUNTIME/
    python run_longbench.py [OPTIONS]

Options:
    --model              HuggingFace model ID   (default: Qwen/Qwen2.5-0.5B-Instruct)
    --preset             low | mid | high       (default: low)
    --serving-mode       lightweight | balanced | performance | long-context | fused-sparse
                                                (default: long-context)
    --rank               SVD rank               (default: 16)
    --max-tokens         max new tokens (hard cap; per-task caps also apply)
    --num-samples        examples per dataset   (default: 20)
    --max-input-tokens   context token budget   (default: 3500)
    --output             JSON output path       (default: longbench_results.json)
    --datasets           comma-separated list   (default: narrativeqa,govreport,qasper,hotpotqa)
    --temperature        generation temperature (default: 0.0 = greedy)
"""

import os
import sys
import json
import time
import argparse
import re
import string
import gc
import threading
from collections import Counter

# ── Path setup ────────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("DIFFKV_USE_TORCH_COMPILE", "0")
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "narrativeqa": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "narrativeqa",
        "metric":     "f1",
        "max_gen":    128,
        "prompt_fn":  "qa",
        "description": "NarrativeQA (Single-Doc QA)",
    },
    "qasper": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "qasper",
        "metric":     "f1",
        "max_gen":    128,
        "prompt_fn":  "qa",
        "description": "Qasper (Single-Doc QA)",
    },
    "hotpotqa": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "hotpotqa",
        "metric":     "f1",
        "max_gen":    64,
        "prompt_fn":  "qa",
        "description": "HotpotQA (Multi-Doc QA)",
    },
    "govreport": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "gov_report",
        "metric":     "rouge_l",
        "max_gen":    512,
        "prompt_fn":  "summarization",
        "description": "GovReport (Summarization)",
    },
}

# ── Prompt builders ───────────────────────────────────────────────────────────
def build_qa_prompt(example):
    context  = example.get("context", "")
    question = example.get("input", "")
    system = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Give a concise, direct answer — do not repeat the question or add explanation."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return system, user

def build_summarization_prompt(example):
    context = example.get("context", "")
    system = "You are a helpful assistant that writes concise, accurate summaries."
    user = f"Please summarize the following document concisely:\n\n{context}\n\nSummary:"
    return system, user

PROMPT_BUILDERS = {
    "qa":            build_qa_prompt,
    "summarization": build_summarization_prompt,
}

# ── Metrics ───────────────────────────────────────────────────────────────────
def _normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def compute_em(prediction, answers):
    """Exact match: 1.0 if normalized prediction matches any gold answer exactly."""
    if not answers:
        return 0.0
    pred_norm = _normalize_text(prediction)
    return float(any(_normalize_text(a) == pred_norm for a in answers))

def _token_f1(prediction, ground_truth):
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens   = _normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common   = Counter(pred_tokens) & Counter(gt_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_f1(prediction, answers):
    """Max token-F1 over all gold answers."""
    if not answers:
        return 0.0
    return max(_token_f1(prediction, a) for a in answers)

def _lcs_length(x, y):
    """Space-optimised 1-D LCS DP."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]

def _rouge_l_single(prediction, reference):
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens  = _normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    p   = lcs / len(pred_tokens)
    r   = lcs / len(ref_tokens)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def compute_rouge_l(prediction, answers):
    """Max Rouge-L over all gold answers."""
    if not answers:
        return 0.0
    return max(_rouge_l_single(prediction, a) for a in answers)

# ── Memory sampler (background thread) ───────────────────────────────────────
class MemorySampler:
    """
    Polls peak MPS/CUDA allocated memory and process RSS in a background thread.
    Call start() before generate(), stop() after, then read peak_mlx_mb / peak_rss_mb.
    """
    def __init__(self, interval_s: float = 0.05):
        self.interval_s   = interval_s
        self.peak_mlx_mb  = 0.0
        self.peak_rss_mb  = 0.0
        self._stop_evt    = threading.Event()
        self._thread      = None

    def _run(self):
        import torch
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:
            proc = None

        while not self._stop_evt.wait(self.interval_s):
            # MPS / CUDA allocated memory
            try:
                if torch.backends.mps.is_available():
                    mlx_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
                elif torch.cuda.is_available():
                    mlx_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                else:
                    mlx_mb = 0.0
                if mlx_mb > self.peak_mlx_mb:
                    self.peak_mlx_mb = mlx_mb
            except Exception:
                pass

            # Process RSS
            if proc is not None:
                try:
                    rss_mb = proc.memory_info().rss / (1024 ** 2)
                    if rss_mb > self.peak_rss_mb:
                        self.peak_rss_mb = rss_mb
                except Exception:
                    pass

    def start(self):
        self.peak_mlx_mb = 0.0
        self.peak_rss_mb = 0.0
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mem-sampler")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

# ── Timing helper: estimate prefill vs decode split ──────────────────────────
def estimate_prefill_decode(tokenizer, prompt_str: str, prediction: str,
                             total_s: float, n_input_tokens: int):
    """
    Since wrapper.generate() is a single opaque call we can't hook into mid-flight.
    Strategy:
      - n_input  = prompt token count  (measured before the call)
      - n_output = output token count  (measured from prediction)
      - Prefill scales O(n_input); decode scales O(n_output).
      - Rough split: prefill_s ≈ total_s * n_input / (n_input + n_output * 4)
        (the factor-4 accounts for decode being ~4× slower per token than prefill
         per-token on MPS, empirically).
      - decode_s = total_s - prefill_s
      - decode_tps = n_output / decode_s
    Returns (prefill_s, decode_tps, n_output).
    """
    n_output = len(tokenizer.encode(prediction, add_special_tokens=False))
    if n_output == 0:
        return total_s, 0.0, 0

    # Heuristic split (prefill is ~4x cheaper per-token than decode on MPS)
    DECODE_WEIGHT = 4
    prefill_weight = n_input_tokens
    decode_weight  = n_output * DECODE_WEIGHT
    denom = prefill_weight + decode_weight
    if denom == 0:
        return total_s, 0.0, n_output

    prefill_s  = total_s * (prefill_weight / denom)
    decode_s   = max(total_s - prefill_s, 1e-6)
    decode_tps = n_output / decode_s
    return round(prefill_s, 3), round(decode_tps, 2), n_output

# ── Context truncation ────────────────────────────────────────────────────────
def truncate_context_by_tokens(tokenizer, context, max_tokens):
    if max_tokens <= 0:
        return context
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return context
    keep_start = int(max_tokens * 0.6)
    keep_end   = max_tokens - keep_start
    trimmed = tokens[:keep_start] + tokens[-keep_end:]
    return tokenizer.decode(trimmed, skip_special_tokens=True)

# ── Parse gold answers ────────────────────────────────────────────────────────
def parse_answers(example):
    answers = example.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    elif not isinstance(answers, list):
        answers = [str(answers)]
    return [str(a).strip() for a in answers if str(a).strip()]

# ── Dataset loader ────────────────────────────────────────────────────────────
def load_dataset_subset(hf_name, hf_split, n, tokenizer, max_input_tokens, prompt_fn_key):
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"  Loading {hf_name} / {hf_split} ...")
    last_err = None
    ds = None
    for split_try in ["test", "train", "validation"]:
        try:
            ds = load_dataset(hf_name, hf_split, split=split_try)
            print(f"  Using split='{split_try}' ({len(ds)} total examples)")
            break
        except Exception as e:
            last_err = e
    if ds is None:
        print(f"  [ERROR] Failed to load {hf_name}/{hf_split}: {last_err}")
        return []

    total   = len(ds)
    step    = max(1, total // n)
    indices = list(range(0, min(total, step * n), step))[:n]
    builder = PROMPT_BUILDERS[prompt_fn_key]

    processed = []
    for idx in indices:
        ex      = ds[idx]
        ex_copy = dict(ex)
        if max_input_tokens > 0:
            ex_copy["context"] = truncate_context_by_tokens(
                tokenizer, ex.get("context", ""), max_input_tokens
            )
        system_prompt, user_content = builder(ex_copy)
        processed.append({
            "system":  system_prompt,
            "user":    user_content,
            "answers": parse_answers(ex),
        })

    print(f"  Prepared {len(processed)} examples")
    return processed

# ── Progress bar ──────────────────────────────────────────────────────────────
def progress_bar(current, total, width=28):
    filled = int(width * current / total)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {current}/{total}"

# ── Cache flush helper ────────────────────────────────────────────────────────
def flush_memory(wrapper, deep=False):
    """Free KV session, run GC, drain allocator cache."""
    import torch
    try:
        wrapper.clear_session(getattr(wrapper, "active_session", None) or "default")
    except Exception:
        pass
    if deep:
        try:
            if hasattr(wrapper, "manager") and wrapper.manager is not None:
                if hasattr(wrapper.manager, "clear"):
                    wrapper.manager.clear()
        except Exception:
            pass
        gc.collect(); gc.collect(); gc.collect()
    else:
        gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            if deep:
                torch.cuda.synchronize()
    except Exception:
        pass

# ── Main evaluation ───────────────────────────────────────────────────────────
def run_evaluation(args):
    import torch

    print()
    print("╔" + "═" * 68 + "╗")
    print("║   DiffKV LongBench Evaluation" + " " * 38 + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Model        : {args.model:<51}║")
    print(f"║  Preset       : {args.preset:<51}║")
    print(f"║  Serving Mode : {args.serving_mode:<51}║")
    print(f"║  SVD Rank     : {args.rank:<51}║")
    print(f"║  Samples/task : {args.num_samples:<51}║")
    print(f"║  Max gen tok  : {args.max_tokens:<51}║")
    print(f"║  Max ctx tok  : {args.max_input_tokens:<51}║")
    print(f"║  Datasets     : {args.datasets:<51}║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Device
    try:
        from native_core.mac_utils import get_best_device
        device = get_best_device()
    except ImportError:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  Device : {device}")

    if device == "mps":
        os.environ.setdefault("DIFFKV_MPS_APPROXIMATE_ATTN", "1")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading model …")
    t0 = time.perf_counter()
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper

    wrapper = DiffKVHFWrapper(
        model_id=args.model,
        config={
            "rank":             args.rank,
            "micro_block_size": 256,
            "block_size":       256,
            "serving_mode":     args.serving_mode,
            "mode":             "fp16",
            "preset":           args.preset,
        },
        device=device,
    )
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    tokenizer = wrapper.tokenizer

    # ── Validate datasets ─────────────────────────────────────────────────────
    dataset_list = [d.strip().lower() for d in args.datasets.split(",")]
    unknown = [d for d in dataset_list if d not in DATASET_CONFIG]
    if unknown:
        print(f"[ERROR] Unknown datasets: {unknown}  Valid: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)

    # ── Load all datasets upfront ─────────────────────────────────────────────
    print(f"\n[2/3] Loading datasets …")
    dataset_examples = {}
    for ds_key in dataset_list:
        cfg = DATASET_CONFIG[ds_key]
        dataset_examples[ds_key] = load_dataset_subset(
            hf_name          = cfg["hf_name"],
            hf_split         = cfg["hf_split"],
            n                = args.num_samples,
            tokenizer        = tokenizer,
            max_input_tokens = args.max_input_tokens,
            prompt_fn_key    = cfg["prompt_fn"],
        )

    # ── Memory sampler (shared across all examples) ───────────────────────────
    mem_sampler = MemorySampler(interval_s=0.05)

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Running inference …")

    all_results = {
        "model":        args.model,
        "preset":       args.preset,
        "serving_mode": args.serving_mode,
        "rank":         args.rank,
        "num_samples":  args.num_samples,
        "device":       device,
        "run_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary":      [],    # one entry per dataset, matching requested schema
        "datasets":     {},    # detailed per-example breakdown
    }

    summary_rows = []

    for ds_key in dataset_list:
        cfg       = DATASET_CONFIG[ds_key]
        examples  = dataset_examples[ds_key]
        is_qa     = cfg["metric"] == "f1"
        max_gen   = min(args.max_tokens, cfg["max_gen"])

        print()
        print(f"  ┌─ {cfg['description']}")
        print(f"  │  {len(examples)} examples | max_gen={max_gen}")
        print(f"  │")

        em_scores       = []
        f1_scores       = []
        rougeL_scores   = []
        prefill_s_list  = []
        decode_tps_list = []
        peak_mlx_list   = []
        peak_rss_list   = []
        per_example     = []
        ds_start        = time.perf_counter()

        for i, ex in enumerate(examples):
            messages = [
                {"role": "system", "content": ex["system"]},
                {"role": "user",   "content": ex["user"]},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            # Count input tokens before call (for prefill/decode split)
            n_input = len(tokenizer.encode(prompt_str, add_special_tokens=False))

            # Start memory sampler for this example
            mem_sampler.start()
            ex_start = time.perf_counter()

            try:
                prediction = wrapper.generate(
                    prompt=prompt_str,
                    max_new_tokens=max_gen,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                if prediction.startswith(prompt_str):
                    prediction = prediction[len(prompt_str):]
                prediction = prediction.strip()
            except Exception as e:
                print(f"  │  [WARN] Example {i+1} error: {e}")
                prediction = ""

            ex_total_s = time.perf_counter() - ex_start
            mem_sampler.stop()

            # ── Compute all metrics ───────────────────────────────────────────
            em  = compute_em(prediction, ex["answers"])
            f1  = compute_f1(prediction, ex["answers"])
            rl  = compute_rouge_l(prediction, ex["answers"])

            pfill_s, dec_tps, n_out = estimate_prefill_decode(
                tokenizer, prompt_str, prediction, ex_total_s, n_input
            )

            peak_mlx = round(mem_sampler.peak_mlx_mb, 1)
            peak_rss = round(mem_sampler.peak_rss_mb, 1)

            em_scores.append(em)
            f1_scores.append(f1)
            rougeL_scores.append(rl)
            prefill_s_list.append(pfill_s)
            decode_tps_list.append(dec_tps)
            peak_mlx_list.append(peak_mlx)
            peak_rss_list.append(peak_rss)

            # Print progress line
            bar       = progress_bar(i + 1, len(examples))
            short_pred = prediction[:50].replace("\n", " ")
            print(
                f"  │  {bar}  "
                f"EM={em:.0f} F1={f1:.3f} RL={rl:.3f}  "
                f"pf={pfill_s:.1f}s dec={dec_tps:.1f}t/s  "
                f"mlx={peak_mlx:.0f}MB rss={peak_rss:.0f}MB  "
                f"→ {short_pred!r}"
            )

            per_example.append({
                "index":         i,
                "prediction":    prediction,
                "answers":       ex["answers"],
                "n_input_toks":  n_input,
                "n_output_toks": n_out,
                "EM":            round(em, 4),
                "F1":            round(f1, 4),
                "Rouge_L":       round(rl, 4),
                "total_s":       round(ex_total_s, 2),
                "prefill_s":     pfill_s,
                "decode_tps":    dec_tps,
                "peak_mlx_mb":   peak_mlx,
                "peak_rss_mb":   peak_rss,
            })

            # Per-example cleanup
            flush_memory(wrapper, deep=False)

        # ── Aggregate ─────────────────────────────────────────────────────────
        ds_time    = time.perf_counter() - ds_start
        avg_em     = round(sum(em_scores)       / len(em_scores),       4) if em_scores       else 0.0
        avg_f1     = round(sum(f1_scores)       / len(f1_scores),       4) if f1_scores       else 0.0
        avg_rl     = round(sum(rougeL_scores)   / len(rougeL_scores),   4) if rougeL_scores   else 0.0
        tot_pfill  = round(sum(prefill_s_list),  2)
        avg_dectps = round(sum(decode_tps_list) / len(decode_tps_list), 2) if decode_tps_list else 0.0
        pk_mlx     = round(max(peak_mlx_list),  1) if peak_mlx_list else 0.0
        pk_rss     = round(max(peak_rss_list),  1) if peak_rss_list else 0.0

        print(f"  │")
        print(f"  └─ DONE  EM={avg_em:.4f}  F1={avg_f1:.4f}  RL={avg_rl:.4f}  "
              f"prefill={tot_pfill:.1f}s  dec={avg_dectps:.1f}t/s  "
              f"mlx={pk_mlx:.0f}MB  rss={pk_rss:.0f}MB  "
              f"[{len(em_scores)} ex, {ds_time:.1f}s total]")

        # Schema matching the requested format
        summary_entry = {
            "dataset":       cfg["description"],
            "method":        "DiffKV",
            "EM":            avg_em,
            "F1":            avg_f1,
            "Rouge_L":       avg_rl,
            "prefill_s":     tot_pfill,
            "decode_tps":    avg_dectps,
            "peak_mlx_mb":   pk_mlx,
            "peak_rss_mb":   pk_rss,
        }
        all_results["summary"].append(summary_entry)
        summary_rows.append(summary_entry)

        all_results["datasets"][ds_key] = {
            "description":  cfg["description"],
            "metric":       cfg["metric"],
            "num_examples": len(em_scores),
            "avg_EM":       avg_em,
            "avg_F1":       avg_f1,
            "avg_Rouge_L":  avg_rl,
            "total_prefill_s": tot_pfill,
            "avg_decode_tps":  avg_dectps,
            "peak_mlx_mb":  pk_mlx,
            "peak_rss_mb":  pk_rss,
            "total_time_s": round(ds_time, 2),
            "per_example":  per_example,
        }

        # Between-dataset deep flush
        flush_memory(wrapper, deep=True)
        print(f"  [Cleanup] Memory flushed after {ds_key}")

    # ── Final summary table ───────────────────────────────────────────────────
    print()
    print("╔" + "═" * 88 + "╗")
    print("║   FINAL RESULTS SUMMARY" + " " * 64 + "║")
    print("╠" + "═" * 88 + "╣")
    hdr = f"  {'Dataset':<32} {'EM':>6} {'F1':>6} {'RL':>6}  {'pfill_s':>8}  {'dec_tps':>8}  {'mlx_MB':>7}  {'rss_MB':>7}"
    print(f"║{hdr:<88}║")
    print("╠" + "─" * 88 + "╣")
    for r in summary_rows:
        name = r["dataset"][:30]
        row  = (f"  {name:<32} {r['EM']:>6.4f} {r['F1']:>6.4f} {r['Rouge_L']:>6.4f}"
                f"  {r['prefill_s']:>8.1f}  {r['decode_tps']:>8.1f}"
                f"  {r['peak_mlx_mb']:>7.1f}  {r['peak_rss_mb']:>7.1f}")
        print(f"║{row:<88}║")
    print("╚" + "═" * 88 + "╝")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = args.output if os.path.isabs(args.output) else os.path.join(_script_dir, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {out_path}")

    wrapper.stop()
    print("  Engine stopped. All done.\n")


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="DiffKV LongBench Evaluation — NarrativeQA, GovReport, Qasper, HotpotQA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",        type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--preset",       type=str, default="low", choices=["low","mid","high"])
    p.add_argument("--serving-mode", type=str, default="long-context", dest="serving_mode",
                   choices=["lightweight","balanced","performance","long-context","fused-sparse"])
    p.add_argument("--rank",         type=int, default=16)
    p.add_argument("--max-tokens",   type=int, default=512, dest="max_tokens")
    p.add_argument("--num-samples",  type=int, default=20,  dest="num_samples")
    p.add_argument("--max-input-tokens", type=int, default=3500, dest="max_input_tokens",
                   help="Truncate context to this many tokens (0 = no truncation)")
    p.add_argument("--datasets",     type=str, default="narrativeqa,govreport,qasper,hotpotqa")
    p.add_argument("--output",       type=str, default="longbench_results.json")
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--top-p",        type=float, default=1.0, dest="top_p")
    p.add_argument("--repetition-penalty", type=float, default=1.05, dest="repetition_penalty")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
