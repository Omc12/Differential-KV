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
from runtime.scaling_integrity_guard import ScalingIntegrityGuard
from runtime.scaling_trace_aggregator import ScalingTraceAggregator, SparseSurvivabilityCurveBuilder

# Semantic Validation Stack
from runtime.semantic_equivalence_validator import SemanticEquivalenceValidator
from runtime.sparse_governance_truth_layer import SparseGovernanceTruthLayer
from runtime.unsafe_suppression_detector import UnsafeSuppressionDetector
from runtime.dense_reference_comparator import DenseReferenceComparator
from runtime.semantic_drift_trace_system import SemanticDriftTraceSystem

# SAG Instruments
from runtime.sparse_confidence_estimator import SparseConfidenceEstimator
from runtime.adaptive_hybrid_suppression_layer import AdaptiveHybridSuppressionLayer
from runtime.layer_semantic_stability_tracker import LayerSemanticStabilityTracker
from runtime.sparse_window_effectiveness_analyzer import SparseWindowEffectivenessAnalyzer
from runtime.sparse_arithmetic_participation_meter import SparseArithmeticParticipationMeter

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ---------------------------------------------------------------------------
# SGC Scaling Configuration (REDUCED FOR SEMANTIC DEPTH)
# ---------------------------------------------------------------------------
SCALING_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "concurrency": 2, "layers": 24, "size_b": 0.5},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "concurrency": 2, "layers": 28, "size_b": 1.5},
    {"id": "Qwen/Qwen2.5-3B-Instruct",   "concurrency": 1, "layers": 36, "size_b": 3.0},
    {"id": "Qwen/Qwen2.5-7B-Instruct",   "concurrency": 1, "layers": 28, "size_b": 7.0},
]

PHASE        = "phase_39_1_sgc"
DURATION_SEC = 120  # 2 minutes per model for deep semantic audit
SAMPLING_INT = 10   # Run dense reference every 10 tokens

PROMPTS = [
    "Identify the logical inconsistency in the following statement: 'The faster I go, the behinder I get.'",
    "Explain the implications of a sparse attention window on long-context reasoning.",
    "Verify the mathematical correctness of the self-attention mechanism in Transformers.",
    "Discuss the trade-offs between sparse persistence and semantic accuracy in LLM inference."
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SGC_SEMANTIC: %(message)s")
logger = logging.getLogger("SGC_Semantic_Validation")

# ---------------------------------------------------------------------------
# Core Session Logic
# ---------------------------------------------------------------------------

async def run_semantic_session(sid, resolver, instruments, semantic_stack, step_counter):
    """
    Execution logic for a single session with Semantic Verification.
    """
    prompt = PROMPTS[hash(sid) % len(PROMPTS)]
    payload = {"session_id": sid, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128}
    
    # Unpack instruments
    conf_est, supp_layer, stab_tracker, win_analyzer, arith_meter = instruments
    # Unpack semantic stack
    validator, truth_layer, unsafe_detector, comparator, tracer = semantic_stack
    
    token_count = 0
    num_layers = len(stab_tracker.stabilized_layers) if hasattr(stab_tracker, 'stabilized_layers') else 24

    try:
        async for chunk in resolver.execute_stream(payload):
            token_count += chunk.get("token_count", 1)
            decode_step = token_count
            input_ids = chunk.get("input_ids") # Assuming resolver returns this for comparison
            
            # 1. Run Governance Stack
            # (In a real system, this happens inside the fusion engine)
            confidence_list = []
            suppressed_list = []
            for layer_idx in range(num_layers):
                confidence = conf_est.record_attention_outcome(
                    step=decode_step, layer_idx=layer_idx, 
                    gate_score=0.85, window_hit_rate=0.92, 
                    fallback_rate=0.05, seq_len=decode_step, mode="sparse"
                )
                decision = supp_layer.evaluate(
                    layer_idx=layer_idx, proposed_mode="hybrid",
                    confidence=confidence, gate_score=0.85,
                    fallback_rate=0.05, step=decode_step
                )
                supp_layer.record_outcome(layer_idx, decision.suppressed, False)
                confidence_list.append(confidence)
                suppressed_list.append(decision.suppressed)

            # 2. SEMANTIC VALIDATION (Periodic)
            if comparator.should_run_reference() and input_ids is not None:
                # Get Sparse Logits (simulated/captured from chunk if available)
                sparse_logits = chunk.get("logits")
                if sparse_logits is not None:
                    # Execute Dense Reference Pass
                    dense_logits = await resolver.run_dense_reference(sid, input_ids)
                    
                    # Calculate Drift
                    drift = validator.calculate_drift(sparse_logits, dense_logits)
                    score = validator.get_semantic_score(drift)
                    is_preserved = validator.is_semantically_correct(drift)
                    
                    # Record Equivalence
                    sparse_token = torch.argmax(sparse_logits, dim=-1).item()
                    dense_token = torch.argmax(dense_logits, dim=-1).item()
                    validator.verify_token_match(sparse_token, dense_token)
                    
                    # Log to Trace System
                    tracer.record_drift(decode_step, drift, score)
                    tracer.record_equivalence(decode_step, sparse_token == dense_token, sparse_token, dense_token)
                    
                    # Check for Unsafe Suppression
                    avg_conf = sum(confidence_list) / len(confidence_list)
                    any_suppressed = any(suppressed_list)
                    unsafe_event = unsafe_detector.evaluate_suppression(any_suppressed, avg_conf, drift)
                    if unsafe_event["is_unsafe_suppression"]:
                        tracer.record_unsafe(decode_step, unsafe_event)
                    
                    # Record Truth
                    truth_layer.record_step(any_suppressed, is_preserved, avg_conf)
            
            step_counter[0] += 1
            
    except Exception as e:
        logger.warning(f"Session {sid} failed: {e}")
    return token_count

# ---------------------------------------------------------------------------
# Scaling Orchestrator
# ---------------------------------------------------------------------------

async def run_sgc_semantic_validation():
    lifecycle = ScalingRuntimeLifecycleManager(phase=PHASE)
    guard = ScalingIntegrityGuard()
    
    # We don't use curve_builder/aggregator yet because we are in the 'TRUTH' phase.
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16") if HAS_BNB else None

    for model_cfg in SCALING_MODELS:
        model_id = model_cfg["id"]
        concurrency = model_cfg["concurrency"]
        num_layers = model_cfg["layers"]
        
        run_mgr = lifecycle.create_run_manager(model_id, concurrency, DURATION_SEC)
        run_mgr.begin()
        
        # 1. Startup
        resolver = await lifecycle.startup_model(model_id, bnb_config, 64, 16)
        
        # 2. Setup Instruments
        instruments = _init_governance_instruments(run_mgr, num_layers)
        semantic_stack = _init_semantic_stack(run_mgr)
        
        step_counter = [0]
        start_time = time.time()
        
        # 3. Launch Concurrent Sessions
        session_ids = [f"sgc-semantic-{run_mgr.run_id}-{i:02d}" for i in range(concurrency)]
        tasks = [asyncio.create_task(run_semantic_session(sid, resolver, instruments, semantic_stack, step_counter)) for sid in session_ids]
        
        # 4. Monitoring Loop
        try:
            while time.time() - start_time < DURATION_SEC:
                if all(t.done() for t in tasks): break
                elapsed = time.time() - start_time
                _print_semantic_status(elapsed, model_id, semantic_stack, step_counter[0])
                await asyncio.sleep(5.0)
        except KeyboardInterrupt:
            break
            
        # 5. Shutdown & Seal
        await lifecycle.shutdown()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 6. Final Truth Recording
        truth_metrics = semantic_stack[1].get_truth_metrics()
        semantic_stack[4].record_truth(truth_metrics)
        
        lifecycle.seal_run({"semantic_metrics": truth_metrics})
        
        # 7. Semantic Integrity Audit
        if guard.validate_run(run_mgr, [
            "semantic_drift_trace.jsonl", "governance_truth_trace.jsonl",
            "sparse_confidence_trace.jsonl"
        ]):
            logger.info(f"Model {model_id} SEMANTIC VALIDATION PASSED.")
        else:
            logger.error(f"Model {model_id} SEMANTIC INTEGRITY FAILED.")

    logger.info("--- SGC PHASE 39.1 SEMANTIC RESET COMPLETE ---")

def _init_governance_instruments(run_mgr, num_layers):
    conf = SparseConfidenceEstimator(num_layers=num_layers, trace_path=run_mgr.trace_path("sparse_confidence_trace.jsonl"))
    supp = AdaptiveHybridSuppressionLayer(num_layers=num_layers, trace_path=run_mgr.trace_path("hybrid_suppression_audit.jsonl"))
    stab = LayerSemanticStabilityTracker(num_layers=num_layers, trace_path=run_mgr.trace_path("layer_semantic_trace.jsonl"))
    win  = SparseWindowEffectivenessAnalyzer(num_layers=num_layers, trace_path=run_mgr.trace_path("window_effectiveness_trace.jsonl"))
    arith = SparseArithmeticParticipationMeter(trace_path=run_mgr.trace_path("arithmetic_governance_trace.jsonl"))
    return [conf, supp, stab, win, arith]

def _init_semantic_stack(run_mgr):
    validator = SemanticEquivalenceValidator(drift_threshold=0.08)
    truth = SparseGovernanceTruthLayer()
    unsafe = UnsafeSuppressionDetector(drift_threshold=0.08)
    comp = DenseReferenceComparator(sampling_interval=SAMPLING_INT)
    trace_dir = Path(run_mgr.trace_path("sparse_confidence_trace.jsonl")).parent
    print(f"\n[DEBUG] Semantic Trace Dir: {trace_dir}")
    tracer = SemanticDriftTraceSystem(trace_dir)
    return [validator, truth, unsafe, comp, tracer]

def _print_semantic_status(elapsed, model_id, semantic_stack, total_steps):
    truth = semantic_stack[1].get_truth_metrics()
    print(
        f"\r[SGC SEMANTIC] {model_id.split('/')[-1]:<12} | {int(elapsed):>3}s | "
        f"drift_acc={truth['semantic_correctness_rate']:.2%} | "
        f"persistence={truth['sparse_persistence_rate']:.2%} | "
        f"steps={total_steps:>5}",
        end="", flush=True
    )

if __name__ == "__main__":
    import torch
    asyncio.run(run_sgc_semantic_validation())
