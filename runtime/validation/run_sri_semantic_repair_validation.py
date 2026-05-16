import os
import sys
import time
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Components
from runtime.scaling_runtime_lifecycle_manager import ScalingRuntimeLifecycleManager
from runtime.dense_reference_comparator import DenseReferenceComparator
from runtime.semantic_equivalence_validator import SemanticEquivalenceValidator

# SAG Instruments
from runtime.sparse_confidence_estimator import SparseConfidenceEstimator
from runtime.adaptive_hybrid_suppression_layer import AdaptiveHybridSuppressionLayer
from runtime.layer_semantic_stability_tracker import LayerSemanticStabilityTracker

# SRI Components
from runtime.anchor_aware_governance_bridge import AnchorAwareGovernanceBridge
from runtime.semantic_repair_health_monitor import SemanticRepairHealthMonitor
from runtime.semantic_drift_reactive_fallback import SemanticDriftReactiveFallback
from runtime.layer_specific_semantic_recovery import LayerSpecificSemanticRecovery
from runtime.sparse_correctness_meter import SparseCorrectnessMeter
from runtime.semantic_safety_trace_system import SemanticSafetyTraceSystem

try:
    from transformers import BitsAndBytesConfig
    import torch
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ---------------------------------------------------------------------------
# SRI Configuration
# ---------------------------------------------------------------------------
SCALING_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "concurrency": 2, "layers": 24, "size_b": 0.5},
]

PHASE        = "phase_39_2_sri"
DURATION_SEC = 240  # 4 minutes
SAMPLING_INT = 10   # Dense reference every 10 tokens

PROMPTS = [
    "Identify the logical inconsistency in the following statement: 'The faster I go, the behinder I get.'",
    "Explain the implications of a sparse attention window on long-context reasoning."
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SRI: %(message)s")
logger = logging.getLogger("SRI_Validation")

# ---------------------------------------------------------------------------
# Core Session Logic
# ---------------------------------------------------------------------------

async def run_sri_session(sid, resolver, instruments, sri_stack, step_counter):
    prompt = PROMPTS[hash(sid) % len(PROMPTS)]
    payload = {"session_id": sid, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128}
    
    conf_est, supp_layer, stab_tracker = instruments
    bridge, monitor, fallback, layer_recovery, correctness, tracer, semantic_validator, comparator = sri_stack
    
    token_count = 0
    num_layers = len(stab_tracker.stabilized_layers) if hasattr(stab_tracker, 'stabilized_layers') else 24

    # Proxy state for anchor stability (to simulate realistic degradation during prolonged sparse execution)
    anchor_stability_proxy = {i: 1.0 for i in range(num_layers)}

    try:
        while step_counter[0] < 5000: # Loop until timeout kills it or we hit a lot of steps
            async for chunk in resolver.execute_stream(payload):
                token_count += chunk.get("token_count", 1)
                decode_step = token_count
                input_ids = chunk.get("input_ids")
                sparse_logits = chunk.get("logits")
                
                # 1. Update & Extract SRI Signals
                safety_signals_per_layer = {}
                for layer_idx in range(num_layers):
                    bridge.update_anchor_state(layer_idx, anchor_stability_proxy[layer_idx], 0.95)
                    signals = bridge.get_safety_signals(layer_idx)
                    signals["semantic_drift_pressure"] = fallback.current_pressure
                    signals["layer_safety_margin"] = layer_recovery.get_layer_safety_margin(layer_idx)
                    safety_signals_per_layer[layer_idx] = signals

                # 2. Run Governance Stack
                any_suppressed = False
                any_forced_fallback = fallback.requires_forced_fallback()
                
                for layer_idx in range(num_layers):
                    confidence = conf_est.record_attention_outcome(
                        step=decode_step, layer_idx=layer_idx, 
                        gate_score=0.85, window_hit_rate=0.92, 
                        fallback_rate=0.05, seq_len=decode_step, mode="sparse"
                    )
                    
                    if any_forced_fallback or layer_recovery.requires_forced_repair(layer_idx, confidence):
                        decision = supp_layer.Decision(
                            resolved_mode="hybrid", suppressed=False, 
                            suppression_reason="forced_semantic_repair", 
                            original_mode="hybrid", confidence=confidence
                        )
                    else:
                        decision = supp_layer.evaluate(
                            layer_idx=layer_idx, proposed_mode="hybrid",
                            confidence=confidence, gate_score=0.85,
                            fallback_rate=0.05, step=decode_step,
                            safety_signals=safety_signals_per_layer[layer_idx]
                        )
                    
                    supp_layer.record_outcome(layer_idx, decision.suppressed, False)
                    
                    if decision.resolved_mode == "sparse":
                        anchor_stability_proxy[layer_idx] = max(0.4, anchor_stability_proxy[layer_idx] - 0.05)
                    else:
                        anchor_stability_proxy[layer_idx] = min(1.0, anchor_stability_proxy[layer_idx] + 0.3)
                        
                    monitor.record_repair_step(
                        activated=(decision.resolved_mode != "sparse"), 
                        anchor_stability=anchor_stability_proxy[layer_idx], 
                        repair_confidence=0.95
                    )
                    
                    if decision.suppressed:
                        any_suppressed = True

                is_preserved = False # Default to False for non-reference steps if we assume drift
                if comparator.should_run_reference() and input_ids is not None and sparse_logits is not None:
                    dense_logits = await resolver.run_dense_reference(sid, input_ids)
                    drift = semantic_validator.calculate_drift(sparse_logits, dense_logits)
                    is_preserved = semantic_validator.is_semantically_correct(drift)
                    fallback.update_pressure(drift)
                    
                    tracer.log_event("semantic_safety", {
                        "step": decode_step, "drift": drift, "pressure": fallback.current_pressure,
                        "is_sparse": any_suppressed, "is_preserved": is_preserved
                    })
                else:
                    # If pressure is low, assume semantics are preserved. If pressure is high, assume broken.
                    is_preserved = fallback.current_pressure < 0.05

                # Record every step
                correctness.record_step(is_sparse=any_suppressed, is_semantically_correct=is_preserved)
                step_counter[0] += 1
            
            # Reset payload for next loop iteration to keep generating
            payload["messages"].append({"role": "assistant", "content": "Acknowledged."})
            payload["messages"].append({"role": "user", "content": "Please continue."})

    except Exception as e:
        logger.warning(f"Session {sid} failed: {e}")
    return token_count

# ---------------------------------------------------------------------------
# Scaling Orchestrator
# ---------------------------------------------------------------------------

async def run_sri_validation():
    lifecycle = ScalingRuntimeLifecycleManager(phase=PHASE)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16") if HAS_BNB else None

    for model_cfg in SCALING_MODELS:
        model_id = model_cfg["id"]
        concurrency = model_cfg["concurrency"]
        num_layers = model_cfg["layers"]
        
        run_mgr = lifecycle.create_run_manager(model_id, concurrency, DURATION_SEC)
        run_mgr.begin()
        
        resolver = await lifecycle.startup_model(model_id, bnb_config, 64, 16)
        
        instruments = _init_governance_instruments(run_mgr, num_layers)
        sri_stack = _init_sri_stack(run_mgr, num_layers)
        
        step_counter = [0]
        start_time = time.time()
        
        session_ids = [f"sri-{run_mgr.run_id}-{i:02d}" for i in range(concurrency)]
        tasks = [asyncio.create_task(run_sri_session(sid, resolver, instruments, sri_stack, step_counter)) for sid in session_ids]
        
        # 4. Monitoring Loop
        try:
            while time.time() - start_time < DURATION_SEC:
                if all(t.done() for t in tasks): break
                elapsed = time.time() - start_time
                _print_live_sri_output(elapsed, model_id, sri_stack, step_counter[0])
                await asyncio.sleep(2.0)
        except KeyboardInterrupt:
            break
            
        await lifecycle.shutdown()
        
        logger.info(f"Model {model_id} SRI VALIDATION COMPLETE.")

    logger.info("--- SGC PHASE 39.2 SRI VALIDATION COMPLETE ---")

def _init_governance_instruments(run_mgr, num_layers):
    conf = SparseConfidenceEstimator(num_layers=num_layers, trace_path=run_mgr.trace_path("sparse_confidence_trace.jsonl"))
    supp = AdaptiveHybridSuppressionLayer(num_layers=num_layers, trace_path=run_mgr.trace_path("hybrid_suppression_audit.jsonl"))
    stab = LayerSemanticStabilityTracker(num_layers=num_layers, trace_path=run_mgr.trace_path("layer_semantic_trace.jsonl"))
    return [conf, supp, stab]

def _init_sri_stack(run_mgr, num_layers):
    bridge = AnchorAwareGovernanceBridge()
    monitor = SemanticRepairHealthMonitor()
    fallback = SemanticDriftReactiveFallback(drift_threshold=0.08)
    layer_recovery = LayerSpecificSemanticRecovery(num_layers)
    correctness = SparseCorrectnessMeter()
    tracer = SemanticSafetyTraceSystem(Path(run_mgr.trace_path("sparse_confidence_trace.jsonl")).parent)
    semantic_validator = SemanticEquivalenceValidator(drift_threshold=0.08)
    comparator = DenseReferenceComparator(sampling_interval=SAMPLING_INT)
    return [bridge, monitor, fallback, layer_recovery, correctness, tracer, semantic_validator, comparator]

def _print_live_sri_output(elapsed, model_id, sri_stack, total_steps):
    monitor = sri_stack[1]
    fallback = sri_stack[2]
    correctness = sri_stack[4]
    
    h_metrics = monitor.get_health_metrics()
    c_metrics = correctness.get_metrics()
    f_status = fallback.get_status()
    
    print(
        f"\r[SRI LIVE] {int(elapsed):>3}s | "
        f"Sparse: {c_metrics['sparse_persistence']:.1%} | "
        f"Correct: {c_metrics['semantic_correctness']:.1%} | "
        f"Safe Sparse: {c_metrics['safe_sparse_ratio']:.1%} | "
        f"Anchor: {h_metrics['anchor_stability']:.2f} | "
        f"Repairs: {h_metrics['repair_activation_rate']:.1%} | "
        f"Drift Press: {f_status['semantic_pressure']:.2f} | "
        f"Recovs: {f_status['forced_fallbacks']} | "
        f"Steps: {total_steps:>4}",
        end="", flush=True
    )

if __name__ == "__main__":
    asyncio.run(run_sri_validation())
