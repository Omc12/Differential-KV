#!/usr/bin/env python3
"""
run_longbench.py — LongBench Evaluation for DiffKV ACTIVE_RUNTIME
==================================================================
Runs 20 examples each from: NarrativeQA, GovReport, Qasper, HotpotQA
Metrics:
  - NarrativeQA  → F1
  - Qasper       → F1
  - HotpotQA     → F1
  - GovReport    → Rouge-L

Usage:
    cd ACTIVE_RUNTIME/
    python run_longbench.py [OPTIONS]

Options:
    --model          HuggingFace model ID  (default: Qwen/Qwen2.5-0.5B-Instruct)
    --preset         low | mid | high      (default: low)
    --serving-mode   lightweight | balanced | performance | long-context | fused-sparse
                                           (default: long-context)
    --rank           SVD rank              (default: 16)
    --max-tokens     max tokens to generate (default: 512)
    --num-samples    examples per dataset  (default: 20)
    --max-input-tokens  truncate prompt context to this many tokens (default: 3500)
    --output         path for JSON results (default: longbench_results.json)
    --datasets       comma-separated list  (default: narrativeqa,govreport,qasper,hotpotqa)
    --temperature    generation temperature (default: 0.0 = greedy)
"""

import os
import sys
import json
import time
import argparse
import re
import string
import gc
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
        "description": "NarrativeQA (Single-Doc QA) — F1",
    },
    "qasper": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "qasper",
        "metric":     "f1",
        "max_gen":    128,
        "prompt_fn":  "qa",
        "description": "Qasper (Single-Doc QA) — F1",
    },
    "hotpotqa": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "hotpotqa",
        "metric":     "f1",
        "max_gen":    64,
        "prompt_fn":  "qa",
        "description": "HotpotQA (Multi-Doc QA) — F1",
    },
    "govreport": {
        "hf_name":    "THUDM/LongBench",
        "hf_split":   "gov_report",
        "metric":     "rouge_l",
        "max_gen":    512,
        "prompt_fn":  "summarization",
        "description": "GovReport (Summarization) — Rouge-L",
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
    text = " ".join(text.split())
    return text

def _token_f1(prediction, ground_truth):
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens   = _normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common    = Counter(pred_tokens) & Counter(gt_tokens)
    n_common  = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_f1(prediction, answers):
    if not answers:
        return 0.0
    return max(_token_f1(prediction, a) for a in answers)

def _lcs_length(x, y):
    m, n = len(x), len(y)
    # space-optimised 1-D DP
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
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_rouge_l(prediction, answers):
    if not answers:
        return 0.0
    return max(_rouge_l_single(prediction, a) for a in answers)

METRIC_FNS = {
    "f1":      compute_f1,
    "rouge_l": compute_rouge_l,
}

# ── Context truncation (keep start + end to minimise "lost-in-middle") ────────
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
        print("[ERROR] 'datasets' package not installed.")
        print("        Run: pip install datasets")
        sys.exit(1)

    print(f"  Loading {hf_name} / {hf_split} ...")
    for split_try in ["test", "train", "validation"]:
        try:
            ds = load_dataset(hf_name, hf_split, split=split_try)
            print(f"  Using split='{split_try}' ({len(ds)} total examples)")
            break
        except Exception as e:
            last_err = e
    else:
        print(f"  [ERROR] Failed to load {hf_name}/{hf_split}: {last_err}")
        return []

    total  = len(ds)
    step   = max(1, total // n)
    indices = list(range(0, min(total, step * n), step))[:n]
    examples = [ds[i] for i in indices]

    builder = PROMPT_BUILDERS[prompt_fn_key]
    processed = []
    for ex in examples:
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

# ── Pretty progress bar ───────────────────────────────────────────────────────
def progress_bar(current, total, width=30):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"

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

    # Auto-detect device
    try:
        from native_core.mac_utils import get_best_device
        device = get_best_device()
    except ImportError:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
        print(f"[ERROR] Unknown datasets: {unknown}")
        print(f"        Valid options: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)

    # ── Load all datasets upfront ─────────────────────────────────────────────
    print(f"\n[2/3] Loading datasets …")
    dataset_examples = {}
    for ds_key in dataset_list:
        cfg = DATASET_CONFIG[ds_key]
        examples = load_dataset_subset(
            hf_name          = cfg["hf_name"],
            hf_split         = cfg["hf_split"],
            n                = args.num_samples,
            tokenizer        = tokenizer,
            max_input_tokens = args.max_input_tokens,
            prompt_fn_key    = cfg["prompt_fn"],
        )
        dataset_examples[ds_key] = examples

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
        "datasets":     {},
    }

    summary_rows = []

    for ds_key in dataset_list:
        cfg       = DATASET_CONFIG[ds_key]
        examples  = dataset_examples[ds_key]
        metric_fn = METRIC_FNS[cfg["metric"]]
        max_gen   = min(args.max_tokens, cfg["max_gen"])

        print()
        print(f"  ┌─ {cfg['description']}")
        print(f"  │  {len(examples)} examples | metric={cfg['metric']} | max_gen={max_gen}")
        print(f"  │")

        scores      = []
        per_example = []
        ds_start    = time.perf_counter()

        for i, ex in enumerate(examples):
            messages = [
                {"role": "system", "content": ex["system"]},
                {"role": "user",   "content": ex["user"]},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            ex_start = time.perf_counter()
            try:
                prediction = wrapper.generate(
                    prompt=prompt_str,
                    max_new_tokens=max_gen,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                )
                # Strip echoed prompt if wrapper returns full sequence
                if prediction.startswith(prompt_str):
                    prediction = prediction[len(prompt_str):]
                prediction = prediction.strip()
            except Exception as e:
                print(f"  │  [WARN] Example {i+1} error: {e}")
                prediction = ""

            ex_time = time.perf_counter() - ex_start
            score   = metric_fn(prediction, ex["answers"])
            scores.append(score)

            bar = progress_bar(i + 1, len(examples))
            short_pred = prediction[:60].replace("\n", " ")
            print(f"  │  {bar}  {cfg['metric'].upper()}={score:.3f}  ({ex_time:.1f}s)  → {short_pred!r}")

            per_example.append({
                "index":      i,
                "prediction": prediction,
                "answers":    ex["answers"],
                "score":      round(score, 4),
                "time_s":     round(ex_time, 2),
            })

            # ── Per-example cleanup ────────────────────────────────────────
            # Explicitly free the KV session so compressed blocks go back to
            # the pool immediately — avoids pool exhaustion over 20 examples.
            try:
                wrapper.clear_session(wrapper.active_session or "default")
            except Exception:
                pass
            gc.collect()
            # Flush MPS / CUDA allocator cache to keep peak RAM bounded
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        ds_time   = time.perf_counter() - ds_start
        avg_score = sum(scores) / len(scores) if scores else 0.0
        summary_rows.append((cfg["description"], cfg["metric"], avg_score, len(scores), ds_time))

        print(f"  │")
        print(f"  └─ DONE  Avg {cfg['metric'].upper()} = {avg_score:.4f}  "
              f"({len(scores)} examples, {ds_time:.1f}s)")

        all_results["datasets"][ds_key] = {
            "description":  cfg["description"],
            "metric":       cfg["metric"],
            "num_examples": len(scores),
            "avg_score":    round(avg_score, 4),
            "total_time_s": round(ds_time, 2),
            "per_example":  per_example,
        }

        # ── Between-dataset deep flush ─────────────────────────────────────
        # After finishing a full dataset, do a heavier sweep: triple GC pass
        # + full allocator drain. This ensures no KV state leaks from one
        # task bleeds into the next (e.g. GovReport → Qasper).
        try:
            if hasattr(wrapper, "manager") and wrapper.manager is not None:
                if hasattr(wrapper.manager, "clear"):
                    wrapper.manager.clear()
        except Exception:
            pass
        gc.collect(); gc.collect(); gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        print(f"  [Cleanup] Memory flushed after {ds_key}")

    # ── Final summary table ───────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print("║   FINAL RESULTS" + " " * 52 + "║")
    print("╠" + "═" * 68 + "╣")
    for desc, metric, score, n, t in summary_rows:
        label = f"{desc}"
        val   = f"{metric.upper()} = {score:.4f}  ({n} ex, {t:.0f}s)"
        print(f"║  {label:<38}  {val:<26}║")
    print("╚" + "═" * 68 + "╝")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(_script_dir, out_path)
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
    p.add_argument("--model",        type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model ID or local path")
    p.add_argument("--preset",       type=str, default="low",
                   choices=["low", "mid", "high"],
                   help="Hardware optimization preset")
    p.add_argument("--serving-mode", type=str, default="long-context",
                   dest="serving_mode",
                   choices=["lightweight","balanced","performance","long-context","fused-sparse"],
                   help="KV cache serving mode")
    p.add_argument("--rank",         type=int, default=16,
                   help="SVD rank for KV compression")
    p.add_argument("--max-tokens",   type=int, default=512, dest="max_tokens",
                   help="Max new tokens to generate (hard cap; per-task caps apply too)")
    p.add_argument("--num-samples",  type=int, default=20, dest="num_samples",
                   help="Number of examples per dataset")
    p.add_argument("--max-input-tokens", type=int, default=3500, dest="max_input_tokens",
                   help="Truncate context to this many tokens (0 = no truncation)")
    p.add_argument("--datasets",     type=str,
                   default="narrativeqa,govreport,qasper,hotpotqa",
                   help="Comma-separated datasets to run")
    p.add_argument("--output",       type=str, default="longbench_results.json",
                   help="Output JSON path (relative to ACTIVE_RUNTIME/ or absolute)")
    p.add_argument("--temperature",  type=float, default=0.0,
                   help="Sampling temperature (0 = greedy decoding)")
    p.add_argument("--top-p",        type=float, default=1.0, dest="top_p",
                   help="Top-p nucleus sampling threshold")
    p.add_argument("--repetition-penalty", type=float, default=1.05,
                   dest="repetition_penalty", help="Repetition penalty")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
