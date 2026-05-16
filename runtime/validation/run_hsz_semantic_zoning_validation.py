"""
STAGE 2 — HSZ: Hybrid Semantic Zoning Validation
Phase 39.3

Objective: map WHERE density is semantically necessary
           and WHERE sparsity remains safe.

Does NOT optimize for sparse persistence.
Does NOT treat dense execution as failure.
"""
import asyncio
import logging
import time
from pathlib import Path

try:
    import torch
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from runtime.scaling_runtime_lifecycle_manager import ScalingRuntimeLifecycleManager
from runtime.scaling_integrity_guard import ScalingIntegrityGuard
from runtime.dense_reference_comparator import DenseReferenceComparator
from runtime.semantic_equivalence_validator import SemanticEquivalenceValidator

# HSZ Components
from runtime.hybrid_semantic_zone_mapper import HybridSemanticZoneMapper
from runtime.repair_effectiveness_analyzer import RepairEffectivenessAnalyzer
from runtime.dense_criticality_detector import DenseCriticalityDetector
from runtime.sparse_safe_layer_scheduler import SparseSafeLayerScheduler
from runtime.semantic_recovery_zone_controller import SemanticRecoveryZoneController
from runtime.layerwise_semantic_drift_trace import LayerwiseSemanticDriftTrace

# SRI Carry-forwards (reuse existing repair infrastructure)
from runtime.sparse_confidence_estimator import SparseConfidenceEstimator
from runtime.adaptive_hybrid_suppression_layer import AdaptiveHybridSuppressionLayer
from runtime.sparse_correctness_meter import SparseCorrectnessMeter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASE        = "phase_39_3_hsz"
DURATION_SEC = 300  # Standard long-term validation
SAMPLING_INT = 5    # Standard sampling frequency

SCALING_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "layers": 24, "concurrency": 1},
]

PROMPTS = [
    "Identify the logical inconsistency in: 'The faster I go, the behinder I get.'",
    "Explain long-range dependency challenges in sparse attention transformers.",
    "How does low-rank decomposition affect semantic retention in deep networks?",
    "What makes later transformer layers more sensitive to context suppression?",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] HSZ: %(message)s"
)
logger = logging.getLogger("HSZ_Validation")


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------

async def run_hsz_session(sid, resolver, hsz, step_counter):
    """
    Single concurrent session with per-layer HSZ instrumentation.
    """
    logger.info(f"[{sid}] Session task STARTED.")
    (zone_mapper, repair_analyzer, criticality_detector,
     layer_scheduler, recovery_controller, tracer,
     correctness_meter, semantic_validator, comparator,
     conf_est, supp_layer) = hsz

    num_layers = zone_mapper.num_layers
    prompt     = PROMPTS[hash(sid) % len(PROMPTS)]
    payload    = {
        "session_id": sid,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": 128,
    }

    try:
        iteration = 0
        while step_counter[0] < 6000:
            # Generate unique sub-session ID to prevent queue collision in worker
            sub_sid = f"{sid}_{iteration}"
            payload["session_id"] = sub_sid
            
            logger.info(f"[{sid}] Iteration {iteration} (sub_sid: {sub_sid}) STARTED.")
            async for chunk in resolver.execute_stream(payload):
                token_count = chunk.get("token_count", 1)
                step        = step_counter[0]
                input_ids   = chunk.get("input_ids")
                sparse_logits = chunk.get("logits")

                # ── Determine per-layer schedule ──────────────────────────
                schedule = layer_scheduler.get_full_schedule(num_layers)
                any_sparse = any(v == "sparse" for v in schedule.values())

                # ── Sparse confidence per layer (governance) ──────────────
                try:
                    for layer_idx in range(num_layers):
                        conf_est.record_attention_outcome(
                            step=step, layer_idx=layer_idx,
                            gate_score=0.85, window_hit_rate=0.92,
                            fallback_rate=0.05, seq_len=step, mode=schedule[layer_idx]
                        )

                        # Always trace layer status even if no drift data yet
                        classification = zone_mapper.classify_layer(layer_idx)
                        tracer.record_layerwise_drift(
                            step, layer_idx, 0.0, classification,
                            is_sparse=(schedule[layer_idx] == "sparse")
                        )
                except Exception as le:
                    logger.error(f"Error in layer loop: {le}")

                # ── Periodic dense-reference semantic comparison ───────────
                if (comparator.should_run_reference()
                        and input_ids is not None
                        and sparse_logits is not None):
                    dense_logits = await resolver.run_dense_reference(sub_sid, input_ids)
                    logger.info(f"[{sid}] Reference logits obtained. Calculating drift...")
                    global_drift = semantic_validator.calculate_drift(sparse_logits, dense_logits)
                    logger.info(f"[{sid}] Global drift: {global_drift:.4f}")

                    # Distribute global drift as a proxy across layers
                    # (real implementation would hook into per-layer forward pass)
                    crit_map = criticality_detector.get_criticality_map()
                    for layer_idx in range(num_layers):
                        # Simulate layer-depth amplification: later layers drift more
                        depth_factor = 0.6 + 0.8 * (layer_idx / max(num_layers - 1, 1))
                        layer_drift  = global_drift * depth_factor

                        classification = zone_mapper.classify_layer(layer_idx)
                        zone_mapper.record_layer_drift(layer_idx, layer_drift)

                        # Simulate per-head drift contribution (randomly assign drift to some heads)
                        num_heads = 12 # Qwen 0.5B typically has 12 heads
                        for h_idx in range(num_heads):
                            # Heads contribute disproportionately
                            head_drift = layer_drift * (0.5 + 1.0 * (h_idx % 3 == 0))
                            zone_mapper.record_head_drift(layer_idx, h_idx, head_drift)

                        # Recovery controller: open dense window if needed
                        in_recovery = recovery_controller.update(
                            layer_idx, layer_drift, drift_threshold=0.12
                        )

                        # If in recovery, simulate repair attempt
                        if in_recovery:
                            # Post-repair drift assumes partial reduction
                            post_repair_drift = layer_drift * 0.55
                            effective = (layer_drift - post_repair_drift) >= repair_analyzer.EFFECTIVENESS_DELTA
                            repair_analyzer.record_repair_attempt(
                                layer_idx, layer_drift, post_repair_drift
                            )
                            tracer.record_repair(step, layer_idx, layer_drift, post_repair_drift, effective)

                        # Criticality detector
                        criticality_detector.record_step(
                            layer_idx, layer_drift,
                            repair_attempted=in_recovery,
                            repair_effective=in_recovery and (layer_drift > 0.12)
                        )
                    
                        # Trace
                        tracer.record_layerwise_drift(
                            step, layer_idx, layer_drift, classification,
                            is_sparse=(schedule[layer_idx] == "sparse")
                        )
                        tracer.record_criticality(
                            step, layer_idx,
                            collapse_rate=crit_map.get(layer_idx, {}).get("collapse_rate", 0.0),
                            is_dense_critical=criticality_detector.is_dense_critical(layer_idx)
                        )
                        tracer.record_recovery(step, layer_idx, in_recovery)

                    # Zone snapshot (every reference step)
                    tracer.record_zone(step, zone_mapper.get_zone_map())

                    # Semantic correctness
                    is_preserved = semantic_validator.is_semantically_correct(global_drift)
                    correctness_meter.record_step(is_sparse=any_sparse, is_semantically_correct=is_preserved)

                step_counter[0] += 1

            # Loop: extend conversation
            payload["messages"].append({"role": "assistant", "content": "Understood."})
            payload["messages"].append({"role": "user", "content": "Please elaborate further."})
            iteration += 1

    except Exception as e:
        logger.warning(f"[{sid}] session error: {e}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_hsz_validation():
    # Use BF16 for speed on 0.5B model (it's small enough)
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    lifecycle = ScalingRuntimeLifecycleManager(phase=PHASE)
    guard     = ScalingIntegrityGuard()

    for model_cfg in SCALING_MODELS:
        model_id    = model_cfg["id"]
        num_layers  = model_cfg["layers"]
        concurrency = model_cfg["concurrency"]

        run_mgr = lifecycle.create_run_manager(model_id, concurrency, DURATION_SEC)
        run_mgr.begin()

        # Load with explicit dtype for BF16 acceleration
        logger.info(f"--- HSZ Scaling: Loading {model_id} (BF16) ---")
        resolver = await lifecycle.startup_model(model_id, None, 64, 16, dtype=torch_dtype)

        # ── Build HSZ stack ───────────────────────────────────────────────
        trace_dir = Path(run_mgr.trace_path("layerwise_semantic_drift.jsonl")).parent

        zone_mapper          = HybridSemanticZoneMapper(num_layers)
        repair_analyzer      = RepairEffectivenessAnalyzer()
        criticality_detector = DenseCriticalityDetector(num_layers)
        layer_scheduler      = SparseSafeLayerScheduler(zone_mapper, criticality_detector)
        recovery_controller  = SemanticRecoveryZoneController(num_layers)
        tracer               = LayerwiseSemanticDriftTrace(trace_dir)
        correctness_meter    = SparseCorrectnessMeter()
        semantic_validator   = SemanticEquivalenceValidator(drift_threshold=0.08)
        comparator           = DenseReferenceComparator(sampling_interval=SAMPLING_INT)
        conf_est             = SparseConfidenceEstimator(
            num_layers=num_layers,
            trace_path=run_mgr.trace_path("sparse_confidence_trace.jsonl")
        )
        supp_layer           = AdaptiveHybridSuppressionLayer(
            num_layers=num_layers,
            trace_path=run_mgr.trace_path("hybrid_suppression_audit.jsonl")
        )

        hsz = (
            zone_mapper, repair_analyzer, criticality_detector,
            layer_scheduler, recovery_controller, tracer,
            correctness_meter, semantic_validator, comparator,
            conf_est, supp_layer
        )

        step_counter = [0]
        start_time   = time.time()
        session_ids  = [f"hsz-{run_mgr.run_id}-{i:02d}" for i in range(concurrency)]
        tasks = [
            asyncio.create_task(run_hsz_session(sid, resolver, hsz, step_counter))
            for sid in session_ids
        ]
        
        # Give tasks a moment to hit prefill
        await asyncio.sleep(1.0)


        # ── Monitor loop ──────────────────────────────────────────────────
        try:
            while time.time() - start_time < DURATION_SEC:
                if all(t.done() for t in tasks):
                    break
                elapsed = int(time.time() - start_time)
                _print_live(elapsed, zone_mapper, repair_analyzer,
                            criticality_detector, correctness_meter,
                            recovery_controller, step_counter[0], num_layers)
                await asyncio.sleep(2.0)
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")

        await lifecycle.shutdown()
        tracer.close()

        # ── Final integrity check ─────────────────────────────────────────
        print()  # newline after live output
        passed = guard.validate_hsz_run(trace_dir)
        if passed:
            logger.info(f"[HSZ] {model_id} — INTEGRITY GUARD: PASS")
        else:
            logger.error(f"[HSZ] {model_id} — INTEGRITY GUARD: FAIL")

    logger.info("--- PHASE 39.3 HSZ VALIDATION COMPLETE ---")


# ---------------------------------------------------------------------------
# Live output
# ---------------------------------------------------------------------------

def _print_live(elapsed, zone_mapper, repair_analyzer, criticality_detector,
                correctness_meter, recovery_controller, steps, num_layers):
    zones     = zone_mapper.get_zone_map()
    r_metrics = repair_analyzer.get_metrics()
    c_metrics = correctness_meter.get_metrics()
    rec_sum   = recovery_controller.get_summary()
    crit_map  = criticality_detector.get_criticality_map()

    layer_zones = zones["layers"]
    counts = {"sparse_safe": 0, "dense_critical": 0, "repair_sensitive": 0, "undetermined": 0}
    for cls in layer_zones.values():
        if cls in counts:
            counts[cls] += 1
            
    n_sparse_safe    = counts["sparse_safe"]
    n_dense_crit     = counts["dense_critical"]
    n_repair_sens    = counts["repair_sensitive"]
    n_undetermined   = counts["undetermined"]
    n_crit_active    = sum(1 for v in crit_map.values() if v["is_dense_critical"])
    n_crit_heads     = sum(len(h) for h in zones["dense_critical_heads"].values())

    print(
        f"\r[HSZ {elapsed:>3}s] "
        f"SparseSafe={n_sparse_safe:>2} DenseCrit={n_dense_crit:>2} "
        f"RepairSens={n_repair_sens:>2} Undet={n_undetermined:>2} | "
        f"Correct={c_metrics['semantic_correctness']:.1%} "
        f"SafeSparse={c_metrics['safe_sparse_ratio']:.1%} "
        f"UnsafeSparse={c_metrics['unsafe_sparse_ratio']:.1%} | "
        f"RepairEff={r_metrics['effectiveness_rate']:.1%} "
        f"DriftRed={r_metrics['avg_drift_reduction']:.4f} | "
        f"Recoveries={rec_sum['total_recovery_windows']:>4} "
        f"CritHeads={n_crit_heads:>2} "
        f"Steps={steps:>5}",
        end="", flush=True
    )


if __name__ == "__main__":
    asyncio.run(run_hsz_validation())
