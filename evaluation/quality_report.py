"""
evaluation/quality_report.py — Phase 2.5 Objective 1

Unified quality report aggregator. Collects results from
PerplexityEvaluator and GenerationDriftEvaluator and produces
structured JSON + text summaries.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.perplexity import PerplexityResult
from evaluation.generation_drift import GenerationResult


@dataclass
class QualityReport:
    model_name:  str
    strategies:  List[str]
    perplexity_results: List[PerplexityResult] = field(default_factory=list)
    drift_results:      List[GenerationResult] = field(default_factory=list)

    def add_perplexity(self, results: List[PerplexityResult]):
        self.perplexity_results.extend(results)

    def add_drift(self, results: List[GenerationResult]):
        self.drift_results.extend(results)

    def summary_by_strategy(self) -> Dict[str, Dict]:
        """Aggregate mean metrics per strategy across all prompts."""
        agg: Dict[str, Dict] = {}

        for r in self.perplexity_results:
            if r.label not in agg:
                agg[r.label] = {
                    "perplexity": [], "log_likelihood": [],
                    "token_agreement": [], "mean_kl_div": [],
                    "compression_ratio": [], "eval_ms": [],
                }
            agg[r.label]["perplexity"].append(r.perplexity)
            agg[r.label]["log_likelihood"].append(r.log_likelihood)
            agg[r.label]["token_agreement"].append(r.token_agreement)
            agg[r.label]["mean_kl_div"].append(r.mean_kl_div)
            agg[r.label]["compression_ratio"].append(r.compression_ratio)
            agg[r.label]["eval_ms"].append(r.eval_ms)

        for r in self.drift_results:
            if r.label not in agg:
                agg[r.label] = {}
            agg[r.label].setdefault("first_divergence", []).append(
                r.first_divergence_token if r.first_divergence_token >= 0 else 999
            )
            agg[r.label].setdefault("token_overlap", []).append(r.token_overlap)
            agg[r.label].setdefault("edit_distance", []).append(r.edit_distance_ratio)

        def mean(lst):
            return round(sum(lst) / len(lst), 5) if lst else None

        result = {}
        for label, metrics in agg.items():
            result[label] = {
                k: mean(v) for k, v in metrics.items()
            }
        return result

    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"  QUALITY REPORT — {self.model_name}")
        print(f"{'='*70}")

        summary = self.summary_by_strategy()

        # Baseline first
        base = summary.get("baseline_fp16", {})
        print(f"\n  Baseline FP16: ppl={base.get('perplexity', '?')}")

        header = (f"  {'Strategy':<20} {'PPL':>8} {'Delta-PPL':>10} "
                  f"{'TokAgree':>10} {'KL-div':>10} {'Ratio':>7}")
        print(f"\n{header}")
        print("  " + "-" * (len(header) - 2))

        base_ppl = base.get("perplexity")
        for label, m in sorted(summary.items()):
            if label == "baseline_fp16":
                continue
            ppl   = m.get("perplexity")
            delta = round(ppl - base_ppl, 4) if ppl and base_ppl else None
            agree = m.get("token_agreement")
            kl    = m.get("mean_kl_div")
            ratio = m.get("compression_ratio")

            print(f"  {label:<20} {ppl or '?':>8}  "
                  f"{f'+{delta}' if delta and delta>=0 else str(delta) or '?':>10}  "
                  f"{agree or '?':>10}  {kl or '?':>10}  {ratio or '?':>7}")

        # Drift
        if self.drift_results:
            print(f"\n  Generation Drift:")
            for label, m in sorted(summary.items()):
                if label == "baseline_fp16":
                    continue
                fd  = m.get("first_divergence")
                ov  = m.get("token_overlap")
                ed  = m.get("edit_distance")
                print(f"  {label:<20} first_div={fd}  overlap={ov}  edit={ed}")

    def save(self, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Full results
        full = {
            "model_name":  self.model_name,
            "strategies":  self.strategies,
            "perplexity":  [r.to_dict() for r in self.perplexity_results],
            "drift":       [r.to_dict() for r in self.drift_results],
            "summary":     self.summary_by_strategy(),
        }
        path = out / f"{self.model_name}_quality_report.json"
        with open(path, "w") as f:
            json.dump(full, f, indent=2)
        print(f"  [OK] Quality report saved -> {path}")
        return path
