"""
experiments/exp_quality_eval.py — Phase 2.5 Objective 1

Downstream model quality evaluation.

Compares:
  - baseline FP16 KV
  - periodic DiffKV (64, 128)
  - EMA-balanced DiffKV
  - rolling-threshold DiffKV

Measures:
  - perplexity delta
  - token agreement
  - generation divergence
  - KL divergence from baseline logits

Usage:
    python experiments/exp_quality_eval.py
    python experiments/exp_quality_eval.py --model gpt2 --context-len 256
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.perplexity import PerplexityEvaluator
from evaluation.generation_drift import GenerationDriftEvaluator
from evaluation.quality_report import QualityReport
from kv_collection.hf_collector import PROMPT_LIBRARY


STRATEGIES = ["periodic_64", "periodic_128", "ema_balanced", "rolling_k3"]


def main(args):
    output_dir = Path("results/quality_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  DOWNSTREAM QUALITY EVALUATION — {args.model.upper()}")
    print(f"  context_len={args.context_len}  eval_tokens={args.eval_tokens}")
    print(f"  strategies: {STRATEGIES}")
    print(f"{'='*70}\n")

    # Build prompts
    prompts, text_types = [], []
    for ttype, plist in PROMPT_LIBRARY.items():
        for p in plist[:args.prompts_per_type]:
            prompts.append(p)
            text_types.append(ttype)

    # Build report
    report = QualityReport(
        model_name=args.model,
        strategies=STRATEGIES,
    )

    # ── Perplexity evaluation ─────────────────────────────────────────────────
    print("[1/2] Perplexity Evaluation")
    ppl_eval = PerplexityEvaluator(
        model_name=args.model,
        device="auto",
        max_context_len=args.context_len,
        eval_tokens=args.eval_tokens,
    )
    ppl_eval.load_model()

    for prompt, ttype in zip(prompts, text_types):
        print(f"\n  Prompt [{ttype}]: {prompt[:60]}...")
        results = ppl_eval.evaluate(
            prompts=[prompt],
            text_types=[ttype],
            strategies=STRATEGIES,
        )
        report.add_perplexity(results)

    # ── Generation drift ──────────────────────────────────────────────────────
    if not args.skip_drift:
        print("\n[2/2] Generation Drift Evaluation")
        drift_eval = GenerationDriftEvaluator(
            model_name=args.model,
            device="auto",
            max_new_tokens=args.drift_tokens,
            max_context_len=args.context_len,
        )
        drift_eval._model     = ppl_eval._model     # reuse loaded model
        drift_eval._tokenizer = ppl_eval._tokenizer

        for prompt, ttype in zip(prompts[:4], text_types[:4]):  # subset for drift
            print(f"\n  Drift [{ttype}]: {prompt[:60]}...")
            drift_results = drift_eval.evaluate(
                prompts=[prompt],
                text_types=[ttype],
                strategies=STRATEGIES,
            )
            report.add_drift(drift_results)

    # ── Print and save ────────────────────────────────────────────────────────
    report.print_summary()
    report_path = report.save(str(output_dir))

    print(f"\n[->] Run visualization/plot_quality_eval.py to visualize results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffKV Quality Evaluation")
    parser.add_argument("--model",            default="gpt2",
                        choices=["gpt2", "gpt2-med", "opt-125m", "tinyllama", "phi2"])
    parser.add_argument("--context-len",      type=int, default=256)
    parser.add_argument("--eval-tokens",      type=int, default=40)
    parser.add_argument("--drift-tokens",     type=int, default=30)
    parser.add_argument("--prompts-per-type", type=int, default=1)
    parser.add_argument("--skip-drift",       action="store_true")
    args = parser.parse_args()
    main(args)
