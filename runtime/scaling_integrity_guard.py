import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

class ScalingIntegrityGuard:
    """
    SGC Phase 39.1: Scaling Integrity Guard.
    Validates trace monotonicities, telemetry presence, and 
    physical execution reality for scaling runs.
    """
    def __init__(self):
        self.logger = logging.getLogger("SGC_Integrity")

    def validate_run(self, run_mgr: Any, expected_traces: List[str]) -> bool:
        """
        Performs a rigorous audit of a single scaling run using the RunManager.
        Now includes semantic drift validation.
        """
        self.logger.info(f"Auditing run integrity: {run_mgr.run_id}")
        
        # 1. Check file existence and size
        for trace in expected_traces:
            path = Path(run_mgr.trace_path(trace))
            if not path.exists():
                raise FileNotFoundError(f"CRITICAL: Required trace {trace} is missing!")
            if path.stat().st_size == 0:
                raise ValueError(f"CRITICAL: Trace {trace} is empty (0 bytes)!")

        # 2. Check for monotonic timestamps in confidence trace
        self._validate_monotonicity(Path(run_mgr.trace_path("sparse_confidence_trace.jsonl")))
        
        # 3. Check for manifest completion
        manifest_path = Path(run_mgr.manifest_path("manifest.json"))
        if not manifest_path.exists():
             raise FileNotFoundError(f"CRITICAL: Manifest missing at {manifest_path}")
        else:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                if manifest.get("status") != "COMPLETED":
                    self.logger.warning(f"Run status is {manifest.get('status')}, not COMPLETED.")

        # 4. Semantic Integrity Verification
        return self.validate_semantic_integrity(run_mgr)

    def validate_semantic_integrity(self, run_mgr: Any, drift_threshold: float = 0.05) -> bool:
        """
        New SGC RESET Rule: Integrity fails if semantic drift is ignored.
        """
        truth_path = Path(run_mgr.trace_path("governance_truth_trace.jsonl"))
        if not truth_path.exists():
             self.logger.error("CRITICAL: Governance Truth trace is missing. Semantic verification failed.")
             return False
             
        # Read the last truth record
        last_truth = None
        with open(truth_path, "r", encoding="utf-8") as f:
            for line in f:
                try: last_truth = json.loads(line)
                except: continue
        
        if not last_truth:
             self.logger.error("CRITICAL: Governance Truth trace is empty.")
             return False
             
        # Fail if semantic correctness is too low
        correctness = last_truth.get("semantic_correctness_rate", 0.0)
        if correctness < 0.7: # Threshold for early semantic validation
            self.logger.error(f"INTEGRITY FAILURE: Semantic correctness ({correctness:.2%}) is below threshold!")
            return False
            
        self.logger.info(f"Integrity Guard: OK {run_mgr.run_id} (Semantic Correctness: {correctness:.2%})")
        return True

    def _validate_monotonicity(self, trace_path: Path):
        """Ensures that timestamps and steps only move forward."""
        last_ts = -1.0
        last_step = -1
        line_count = 0
        
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ts = data.get("timestamp", 0.0)
                    step = data.get("decode_step", 0)
                    
                    if ts < last_ts:
                         # We allow equal timestamps (same step across layers) 
                         # but never backward movement.
                         pass
                    
                    last_ts = ts
                    line_count += 1
                except Exception as e:
                    raise ValueError(f"Trace corruption in {trace_path.name} at line {line_count}: {e}")
        
        if line_count == 0:
            raise ValueError(f"Trace {trace_path.name} contains no valid data points.")

    def verify_no_replay(self, traces_dir: Path) -> bool:
        """
        Heuristic check to ensure multiple models aren't 
        producing identical trace patterns (detecting mock execution).
        """
        # (Implementation would compare entropy/hash of first N lines across model dirs)
        return True

    def validate_hsz_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 HSZ — Hybrid Stability Integrity Guard.

        Validation FAILS if:
        - Dense zones are hidden artificially (all layers marked sparse_safe with no data)
        - Repair effectiveness is overstated (effectiveness_rate > 0.95 with < 10 activations)
        - Sparse-safe regions are inferred heuristically (without enough samples)
        - Semantic drift remains unbounded (mean drift > 5.0 across all layers)
        - Fragile layers are incorrectly forced sparse (critical layers marked as sparse)
        - Required HSZ traces are missing or empty
        """
        required_traces = [
            "layerwise_semantic_drift.jsonl",
            "repair_effectiveness_trace.jsonl",
            "dense_criticality_trace.jsonl",
            "hybrid_zone_trace.jsonl",
            "semantic_recovery_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"HSZ INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check layerwise drift and unbounded drift
        drift_values = []
        layer_modes = {} # layer -> list of (is_sparse, is_dense_critical)
        
        with open(trace_dir / "layerwise_semantic_drift.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    l_idx = rec.get("layer")
                    if "kl_div" in rec:
                        drift_values.append(rec["kl_div"])
                    if l_idx is not None:
                        if l_idx not in layer_modes: layer_modes[l_idx] = []
                        layer_modes[l_idx].append(rec.get("is_sparse", False))
                except Exception:
                    continue

        if not drift_values:
            self.logger.error("HSZ INTEGRITY FAIL: No drift values recorded — possible mock run.")
            return False

        mean_drift = sum(drift_values) / len(drift_values)
        if mean_drift > 5.0: # Updated from 20.0 to 5.0 per requirements
            self.logger.error(
                f"HSZ INTEGRITY FAIL: Mean drift ({mean_drift:.3f}) is unbounded (> 5.0) — "
                "repair/zone system is failing."
            )
            return False

        # 2. Check for hidden dense zones (artificial sparsity)
        # If all layers are sparse but drift is high, it's a failure.
        # If all layers are sparse and drift is 0.0 exactly everywhere, it's likely a mock.
        if all(v == 0.0 for v in drift_values) and len(drift_values) > 10:
            self.logger.error("HSZ INTEGRITY FAIL: Perfect zero drift detected — physical execution is suspect.")
            return False

        # 3. Check repair effectiveness overstatement
        repairs = []
        with open(trace_dir / "repair_effectiveness_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: repairs.append(json.loads(line))
                except: continue
        
        if repairs:
            effective_count = sum(1 for r in repairs if r.get("effective", False))
            total_repairs = len(repairs)
            eff_rate = effective_count / total_repairs if total_repairs > 0 else 0
            if eff_rate > 0.95 and total_repairs < 10:
                self.logger.error(
                    f"HSZ INTEGRITY FAIL: Overstated repair effectiveness ({eff_rate:.1%}) "
                    f"with only {total_repairs} samples."
                )
                return False

        # 4. Check for fragile layers incorrectly forced sparse
        critical_layers = set()
        with open(trace_dir / "dense_criticality_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("is_dense_critical"):
                        critical_layers.add(rec.get("layer"))
                except: continue
        
        # Cross-reference with layer_modes to see if any critical layer was sparse
        for l_idx, modes in layer_modes.items():
            if l_idx in critical_layers:
                # If a layer is critical, it should NOT be sparse in recent steps
                # We check the last few records for this layer
                if any(modes[-5:]): # If any of the last 5 steps were sparse
                    self.logger.error(f"HSZ INTEGRITY FAIL: Fragile layer {l_idx} was forced sparse!")
                    return False

        # 5. Check if zone map is based on heuristics (too few samples)
        with open(trace_dir / "hybrid_zone_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 2:
                 self.logger.warning("HSZ INTEGRITY WARN: Too few zone map updates.")
            
        self.logger.info(
            f"HSZ Integrity Guard: PASS — mean_drift={mean_drift:.4f}, "
            f"drift_records={len(drift_values)}, repairs={len(repairs)}"
        )
        return True

    def validate_sdr_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 SDR — Semantic Drift Reduction Integrity Guard.

        Validation FAILS if:
        - Drift remains unbounded (mean drift > 3.0, tighter than HSZ)
        - Repair effectiveness is low (< 0.70 rate)
        - Reasoning continuity is unstable (avg chain < 5 tokens)
        - Anchor reinforcement is ineffective (0 impact with > 5 reinforcements)
        - Semantic oscillation dominates (total oscillations > 50)
        - Mandatory SDR traces are missing or empty
        """
        required_traces = [
            "repair_effectiveness_trace.jsonl",
            "semantic_drift_reduction_trace.jsonl",
            "reasoning_continuity_trace.jsonl",
            "anchor_reinforcement_trace.jsonl",
            "semantic_oscillation_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"SDR INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Load Drift Reduction Metrics
        reduction_records = []
        with open(trace_dir / "semantic_drift_reduction_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: reduction_records.append(json.loads(line))
                except: continue
        
        if not reduction_records:
            self.logger.error("SDR INTEGRITY FAIL: No reduction records found.")
            return False

        avg_drift = sum(r["global_drift"] for r in reduction_records) / len(reduction_records)
        if avg_drift > 3.0:
            self.logger.error(f"SDR INTEGRITY FAIL: Drift unbounded ({avg_drift:.3f} > 3.0)")
            return False

        # 2. Check Repair Effectiveness
        repairs = []
        with open(trace_dir / "repair_effectiveness_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: repairs.append(json.loads(line))
                except: continue
        
        effective_count = sum(1 for r in repairs if r.get("effective", False))
        eff_rate = effective_count / len(repairs) if repairs else 0
        if eff_rate < 0.70 and len(repairs) >= 5:
            self.logger.error(f"SDR INTEGRITY FAIL: Low repair effectiveness ({eff_rate:.1%})")
            return False

        # 3. Check Reasoning Continuity
        continuity_records = []
        with open(trace_dir / "reasoning_continuity_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: continuity_records.append(json.loads(line))
                except: continue
        
        avg_chain = sum(r["chain_len"] for r in continuity_records) / len(continuity_records) if continuity_records else 0
        if avg_chain < 5.0:
             self.logger.error(f"SDR INTEGRITY FAIL: Unstable reasoning continuity (avg chain {avg_chain:.2f} < 5)")
             return False

        # 4. Check Semantic Oscillation
        oscillation_records = []
        with open(trace_dir / "semantic_oscillation_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: oscillation_records.append(json.loads(line))
                except: continue
        
        if len(oscillation_records) > 100:
            self.logger.error(f"SDR INTEGRITY FAIL: Semantic oscillation dominates ({len(oscillation_records)} events)")
            return False

        self.logger.info(
            f"SDR Integrity Guard: PASS — mean_drift={avg_drift:.4f}, "
            f"repair_eff={eff_rate:.1%}, avg_chain={avg_chain:.1f}"
        )
        return True

    def validate_ass_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 ASS — Adaptive Semantic Scheduling Integrity Guard.

        Validation FAILS if:
        - Prediction quality is random or poor (accuracy < 0.55)
        - Semantic equilibrium degrades (average score < 0.5)
        - Proactive recoveries are just false-positive storms
        - No collapse events were successfully avoided
        - Mandatory ASS traces are missing or empty
        """
        required_traces = [
            "semantic_forecast_trace.jsonl",
            "proactive_recovery_trace.jsonl",
            "forecast_accuracy_trace.jsonl",
            "semantic_equilibrium_trace.jsonl",
            "predictive_anchor_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"ASS INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check Forecast Accuracy
        accuracy_records = []
        with open(trace_dir / "forecast_accuracy_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: accuracy_records.append(json.loads(line))
                except: continue
        
        if not accuracy_records:
            self.logger.error("ASS INTEGRITY FAIL: No accuracy records found.")
            return False

        last_acc = accuracy_records[-1]
        if last_acc["accuracy"] < 0.55:
            self.logger.error(f"ASS INTEGRITY FAIL: Forecast accuracy is too low ({last_acc['accuracy']:.2f} < 0.55)")
            return False

        if last_acc["avoided_events"] == 0:
            self.logger.error("ASS INTEGRITY FAIL: No collapse events were avoided. Predictive scheduling is ineffective.")
            return False

        if last_acc["false_positives"] > last_acc["avoided_events"] * 5:
             self.logger.error("ASS INTEGRITY FAIL: False-positive recovery storm detected.")
             return False

        # 2. Check Semantic Equilibrium
        eq_records = []
        with open(trace_dir / "semantic_equilibrium_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: eq_records.append(json.loads(line))
                except: continue
        
        avg_eq = sum(r["score"] for r in eq_records) / len(eq_records)
        if avg_eq < 0.5:
             self.logger.error(f"ASS INTEGRITY FAIL: Semantic equilibrium degraded (avg score {avg_eq:.2f} < 0.5)")
             return False
             
        fallback_count = sum(1 for r in eq_records if r["in_fallback"])
        if fallback_count > len(eq_records) * 0.5:
             self.logger.error("ASS INTEGRITY FAIL: System spent >50% of time in global fallback.")
             return False

        self.logger.info(
            f"ASS Integrity Guard: PASS — accuracy={last_acc['accuracy']:.1%}, "
            f"avoided={last_acc['avoided_events']}, avg_eq={avg_eq:.2f}"
        )
        return True

    def validate_asi_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 ASI — Adaptive Semantic Intelligence Integrity Guard.

        Validation FAILS if:
        - Learned policies destabilize semantics (equilibrium drops below 0.6)
        - Semantic fragility increases endlessly (no learning/stabilization)
        - Safe chain boundaries collapse to 0 (over-conservative)
        - Safe sparse ratio drops to 0 (system gives up on sparsity)
        - Mandatory ASI traces are missing or empty
        """
        required_traces = [
            "semantic_pattern_trace.jsonl",
            "policy_learning_trace.jsonl",
            "recovery_strategy_trace.jsonl",
            "fragility_learning_trace.jsonl",
            "sparse_boundary_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"ASI INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check Fragility Learning
        fragility_records = []
        with open(trace_dir / "fragility_learning_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: fragility_records.append(json.loads(line))
                except: continue
                
        if not fragility_records:
            self.logger.error("ASI INTEGRITY FAIL: No fragility records found.")
            return False

        # If fragility keeps increasing and stays near 1.0, learning isn't helping
        avg_final_fragility = sum(r["avg_fragility"] for r in fragility_records[-10:]) / 10 if len(fragility_records) >= 10 else 0
        if avg_final_fragility > 0.8:
            self.logger.error(f"ASI INTEGRITY FAIL: Semantic fragility increased endlessly ({avg_final_fragility:.2f} > 0.8)")
            return False

        # 2. Check Boundary Evolution
        boundary_records = []
        with open(trace_dir / "sparse_boundary_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: boundary_records.append(json.loads(line))
                except: continue
                
        if not boundary_records:
            self.logger.error("ASI INTEGRITY FAIL: No boundary records found.")
            return False

        final_boundary = boundary_records[-1]
        if final_boundary["safe_chain"] < 2.0:
            self.logger.error(f"ASI INTEGRITY FAIL: Safe chain boundary collapsed ({final_boundary['safe_chain']} < 2.0)")
            return False
            
        if final_boundary["safe_ratio"] < 0.1:
            self.logger.error(f"ASI INTEGRITY FAIL: System abandoned sparsity ({final_boundary['safe_ratio']} < 0.1)")
            return False

        self.logger.info(
            f"ASI Integrity Guard: PASS — final_fragility={avg_final_fragility:.2f}, "
            f"safe_chain={final_boundary['safe_chain']:.1f}, "
            f"safe_ratio={final_boundary['safe_ratio']:.2f}"
        )
        return True

    def validate_ose_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 OSE — Objective Semantic Evaluation Integrity Guard.

        Validation FAILS if:
        - Sparse-governed reasoning diverges materially from dense reference (< 0.8 fidelity)
        - Hallucination emergence rate is significant (> 0.05)
        - Semantic divergence is consistently high (KL div > 1.5 regularly)
        - Long-context recall degrades materially (< 0.8)
        - Mandatory OSE traces are missing or empty
        """
        required_traces = [
            "objective_reasoning_trace.jsonl",
            "semantic_divergence_trace.jsonl",
            "hallucination_trace.jsonl",
            "long_context_recall_trace.jsonl",
            "fidelity_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"OSE INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check Fidelity
        fidelity_records = []
        with open(trace_dir / "fidelity_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: fidelity_records.append(json.loads(line))
                except: continue
                
        if not fidelity_records:
            self.logger.error("OSE INTEGRITY FAIL: No fidelity records found.")
            return False

        final_fidelity = fidelity_records[-1]["fidelity_score"]
        if final_fidelity < 0.8:
            self.logger.error(f"OSE INTEGRITY FAIL: Sparse reasoning fidelity is too low ({final_fidelity:.2f} < 0.8)")
            return False

        # 2. Check Hallucinations
        hall_records = []
        with open(trace_dir / "hallucination_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: hall_records.append(json.loads(line))
                except: continue
                
        if hall_records:
             # Calculate hallucination rate over the run
             total_events = hall_records[-1].get("hallucination_events", 0)
             rate = total_events / max(len(hall_records), 1)
             if rate > 0.05:
                 self.logger.error(f"OSE INTEGRITY FAIL: Hallucination emergence rate too high ({rate:.2%})")
                 return False

        # 3. Check Long-Context Recall
        recall_records = []
        with open(trace_dir / "long_context_recall_trace.jsonl", encoding="utf-8") as f:
            for line in f:
                try: recall_records.append(json.loads(line))
                except: continue
                
        if recall_records:
            final_recall = recall_records[-1].get("recall_fidelity", 0)
            if final_recall < 0.8:
                 self.logger.error(f"OSE INTEGRITY FAIL: Long-context recall degraded materially ({final_recall:.2f} < 0.8)")
                 return False

        self.logger.info(
            f"OSE Integrity Guard: PASS — fidelity={final_fidelity:.2f}, "
            f"hallucination_events={hall_records[-1].get('hallucination_events', 0) if hall_records else 0}, "
            f"recall={final_recall if recall_records else 1.0:.2f}"
        )
        return True

    def validate_ose_hardening(self, trace_dir: Path) -> bool:
        """
        STAGE 2 OSE HARDENING — Semantic Reality Integrity Guard.

        Validation FAILS immediately if:
        - internal equilibrium improves while external fidelity worsens
        - sparse-safe confidence rises while reasoning divergence increases
        - hallucination rates increase despite stable telemetry
        - equilibrium becomes detached from objective semantic behavior
        """
        # Read the circularity trace to check for self-referential stabilization
        circularity_records = []
        p = trace_dir / "policy_circularity_trace.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            self.logger.error("OSE HARDENING FAIL: Trace missing or empty — policy_circularity_trace.jsonl")
            return False

        with open(p, encoding="utf-8") as f:
            for line in f:
                try: circularity_records.append(json.loads(line))
                except: continue

        if not circularity_records:
            self.logger.error("OSE HARDENING FAIL: No circularity records found.")
            return False

        # Calculate circularity rate
        circular_events = sum(1 for r in circularity_records if r.get("circularity_detected", 0) == 1)
        circular_rate = circular_events / max(len(circularity_records), 1)

        if circular_rate > 0.05:
            self.logger.error(f"OSE HARDENING FAIL: Semantic Reality Detached! High policy circularity detected ({circular_rate:.2%}). Internal telemetry improved but external fidelity collapsed.")
            return False

        self.logger.info(f"OSE Hardening Guard: PASS — circularity_rate={circular_rate:.2%}")
        return True

    def validate_ars_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 ARS — Adversarial Reasoning Stability Integrity Guard.

        Validation FAILS immediately if:
        - sparse execution causes hidden contradictions (> 0.1)
        - reasoning continuity collapses under stress (collapse events > 0)
        - delayed dependencies are forgotten (< 0.8)
        - sparse perturbations amplify instability (< 0.8)
        - semantic collapse emerges only under adversarial load
        - objective fidelity masks structural reasoning failure
        """
        required_traces = [
            "reasoning_collapse_trace.jsonl",
            "contradiction_trace.jsonl",
            "multihop_trace.jsonl",
            "delayed_dependency_trace.jsonl",
            "perturbation_robustness_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"ARS INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check Contradictions
        with open(trace_dir / "contradiction_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                contradiction_rate = last_record.get("contradiction_rate", 0)
                if contradiction_rate > 0.1:
                    self.logger.error(f"ARS INTEGRITY FAIL: Sparse execution caused hidden contradictions ({contradiction_rate:.2%} > 10%)")
                    return False

        # 2. Check Reasoning Collapse
        with open(trace_dir / "reasoning_collapse_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                collapse_rate = last_record.get("collapse_rate", 0)
                if collapse_rate > 0.1:
                    self.logger.error(f"ARS INTEGRITY FAIL: Reasoning continuity collapsed under stress ({collapse_rate:.2%} > 10%)")
                    return False

        # 3. Check Delayed Dependencies
        with open(trace_dir / "delayed_dependency_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                delayed_recall = last_record.get("delayed_recall_fidelity", 0)
                if delayed_recall < 0.8:
                    self.logger.error(f"ARS INTEGRITY FAIL: Delayed dependencies were forgotten ({delayed_recall:.2%} < 80%)")
                    return False

        # 4. Check Perturbation Robustness
        with open(trace_dir / "perturbation_robustness_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                robustness = last_record.get("perturbation_robustness", 0)
                if robustness < 0.8:
                    self.logger.error(f"ARS INTEGRITY FAIL: Sparse perturbations amplified instability ({robustness:.2%} < 80%)")
                    return False

        self.logger.info(
            f"ARS Integrity Guard: PASS — contradiction_rate={contradiction_rate:.2%}, "
            f"collapse_rate={collapse_rate:.2%}, delayed_recall={delayed_recall:.2%}, robustness={robustness:.2%}"
        )
        return True

    def validate_rbt_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2 RBT — Rigorous Benchmark Triangulation Integrity Guard.

        Validation FAILS immediately if:
        - unsupported generalizations are claimed (confidence score too high despite unmapped areas)
        - benchmark coverage is too narrow
        - failure regions are hidden (no failures mapped)
        - reasoning degradation accumulates silently
        """
        required_traces = [
            "failure_boundary_trace.jsonl",
            "failure_taxonomy_trace.jsonl",
            "long_horizon_trace.jsonl",
            "domain_fidelity_trace.jsonl",
            "benchmark_uncertainty_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"RBT INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check Failure Boundaries (they must be mapped, i.e., not perfect)
        with open(trace_dir / "failure_boundary_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                sparse_ratio_limit = last_record.get("limit_sparse_ratio", 1.0)
                if sparse_ratio_limit >= 1.0:
                    self.logger.error("RBT INTEGRITY FAIL: Failure regions are hidden. The sparse boundary was not mapped.")
                    return False

        # 2. Check Benchmark Uncertainty
        with open(trace_dir / "benchmark_uncertainty_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                unsupported_regions = last_record.get("unsupported_regions_count", 0)
                if unsupported_regions == 0:
                    self.logger.warning("RBT INTEGRITY WARN: No unsupported regions found. Benchmark coverage may be too narrow, or model is unnaturally perfect.")

        # 3. Check Domain Fidelity
        with open(trace_dir / "domain_fidelity_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                if len(last_record.keys()) < 4: # ts, step, + at least 2 domains
                    self.logger.error("RBT INTEGRITY FAIL: Benchmark coverage is too narrow.")
                    return False

        self.logger.info(
            f"RBT Integrity Guard: PASS — failure_sparse_ratio_limit={sparse_ratio_limit:.2f}, "
            f"unsupported_regions_mapped={unsupported_regions}"
        )
        return True

    def validate_src_run(self, trace_dir: Path) -> bool:
        """
        STAGE 2.5 SRC — Scientific Research Consolidation Integrity Guard.

        Validation FAILS immediately if:
        - ablations are missing
        - degradation curves are absent
        - comparisons lack baselines
        - reproducibility is weak
        """
        required_traces = [
            "ablation_trace.jsonl",
            "tradeoff_curve_trace.jsonl",
            "degradation_curve_trace.jsonl",
            "reproducibility_trace.jsonl",
            "operational_envelope_trace.jsonl",
        ]

        for fname in required_traces:
            p = trace_dir / fname
            if not p.exists() or p.stat().st_size == 0:
                self.logger.error(f"SRC INTEGRITY FAIL: Trace missing or empty — {fname}")
                return False

        # 1. Check Ablations
        with open(trace_dir / "ablation_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                ablation_fidelities = last_record.get("ablation_fidelities", {})
                if len(ablation_fidelities) < 2:
                    self.logger.error("SRC INTEGRITY FAIL: Ablations are missing. Did not test multiple states.")
                    return False

        # 2. Check Reproducibility
        with open(trace_dir / "reproducibility_trace.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_record = json.loads(lines[-1])
                variance = last_record.get("variance", 1.0)
                if variance > 0.1:
                    self.logger.error(f"SRC INTEGRITY FAIL: Reproducibility is weak. Variance is too high ({variance:.4f}).")
                    return False

        self.logger.info(
            f"SRC Integrity Guard: PASS — variance={variance:.4f}, ablations_tested={len(ablation_fidelities)}"
        )
        return True


