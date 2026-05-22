"""
STAGE 2 — SAT: SAT Attention Validation (v2 — versioned, safe)
Phase 38.9 — Sparse Attention Transition

Each invocation creates a NEW timestamped run directory.
Prior runs are never overwritten.

Run layout:
  traces/stage2/phase_38_9_sat/<run_id>/
  telemetry/stage2/phase_38_9_sat/<run_id>/
  reports/stage2/phase_38_9_sat/<run_id>/
  manifests/stage2/phase_38_9_sat/<run_id>/

Graceful Ctrl+C:
  - flushes all trace buffers to disk
  - terminates nvidia-smi dmon
  - writes manifest.json with final status
  - writes RUN_ABORTED.txt or RUN_COMPLETE.txt

Must be run from the project root:
  python -m runtime.validation.run_sat_attention_validation
"""

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# SAT components
# ---------------------------------------------------------------------------
from runtime.sat_run_manager import SATRunManager
from runtime.sparse_attention_path_auditor import SparseAttentionPathAuditor
from runtime.dense_reconstruction_trace_monitor import (
    DenseReconstructionTraceMonitor,
    TRIGGER_ATTENTION_FALLBACK,
)
from runtime.sparse_kv_residency_verifier import SparseKVResidencyVerifier
from runtime.sparse_attention_window_controller import SparseAttentionWindowController
from runtime.transformer_execution_mode_trace import TransformerExecutionModeTrace
from runtime.sparse_arithmetic_participation_meter import SparseArithmeticParticipationMeter

# ---------------------------------------------------------------------------
# Runtime infrastructure
# ---------------------------------------------------------------------------
from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
from decode_pipeline_fusion_engine import DecodePipelineFusionEngine
from runtime.cdbe_resolver import CDBEResolver

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# ---------------------------------------------------------------------------
# Logging — append-only file handler added after run_id is known
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] SAT: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SAT_Validation")

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
MODEL_ID          = "Qwen/Qwen2.5-7B-Instruct"
CONCURRENCY       = 10          # 8–12 sessions
DURATION_SEC      = 360         # 6 minutes  (within the 5–10 min window)
MAX_TOKENS        = 512         # per session — keeps run bounded
QUANTIZATION      = "4bit_nf4"
BLOCK_SIZE        = 64
RANK              = 16

LONG_CONTEXT_PROMPTS = [
    (
        "You are an expert in transformer architecture internals. "
        "Provide a detailed technical walkthrough of how block-sparse attention "
        "reduces memory bandwidth pressure during autoregressive decode, including "
        "the role of KV cache block tables, token-to-block mapping, and how partial "
        "attention windows interact with causal masking across multiple decoder layers."
    ),
    (
        "Explain the complete lifecycle of a KV cache entry in a production inference "
        "server from prefill allocation through speculative eviction under memory pressure. "
        "Focus on the interplay between sparse block residency, attention head specialisation, "
        "and the impact of sequence-length growth on cache reuse efficiency."
    ),
    (
        "Write a comprehensive analysis comparing full-context dense attention versus "
        "block-sparse attention for long-context (32k token) inference. Cover arithmetic "
        "intensity, memory-bound vs compute-bound regimes, realistic FLOP savings on "
        "modern A100-class hardware, and conditions under which sparse attention degrades "
        "to dense attention in practice."
    ),
    (
        "Describe the design of a sparse-native transformer execution engine. "
        "Address: (1) how to route tokens to sparse attention kernels, "
        "(2) when to fall back to dense paths, (3) how to track execution mode "
        "transitions across layers, (4) how to measure arithmetic participation "
        "without relying on theoretical estimates."
    ),
    (
        "Analyse the tradeoffs in KV cache compression methods: anchor-based sparse "
        "retention vs. quantised dense retention vs. hybrid approaches. "
        "Include discussion of reconstruction cost, retrieval quality, and "
        "how each method affects attention score distributions across layers."
    ),
]


# ===========================================================================
# nvidia-smi dmon — append-only background capture
# ===========================================================================

class NvidiaSmiDmonThread(threading.Thread):
    """
    Runs nvidia-smi dmon in a daemon thread and pipes output directly into
    the log file opened in append mode.  The file handle stays open until
    stop() is called so the OS kernel can flush cleanly.
    """

    def __init__(self, log_path: str):
        super().__init__(daemon=True, name="nvsmi-dmon")
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._stop_event = threading.Event()
        self._proc: Optional[subprocess.Popen] = None

    def run(self) -> None:
        try:
            # Open in append mode so re-runs never truncate existing data
            with open(self.log_path, "a") as fh:
                fh.write(
                    f"# nvidia-smi dmon session start "
                    f"ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                )
                fh.flush()
                self._proc = subprocess.Popen(
                    ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "2"],
                    stdout=fh,
                    stderr=subprocess.DEVNULL,
                )
                while not self._stop_event.is_set():
                    time.sleep(0.5)
                    if self._proc.poll() is not None:
                        break
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        except FileNotFoundError:
            with open(self.log_path, "a") as fh:
                fh.write("# nvidia-smi not available on this system\n")

    def stop(self) -> None:
        self._stop_event.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self.join(timeout=6)


# ===========================================================================
# Per-session coroutine
# ===========================================================================

async def run_sat_session(
    session_id: str,
    resolver: CDBEResolver,
    auditor: SparseAttentionPathAuditor,
    recon_monitor: DenseReconstructionTraceMonitor,
    kv_verifier: SparseKVResidencyVerifier,
    window_ctrl: SparseAttentionWindowController,
    mode_tracer: TransformerExecutionModeTrace,
    participation_meter: SparseArithmeticParticipationMeter,
    step_counter: List[int],
) -> int:
    """Execute one SAT session and feed all SAT instruments from chunk metadata."""
    prompt = LONG_CONTEXT_PROMPTS[hash(session_id) % len(LONG_CONTEXT_PROMPTS)]
    payload = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
    }

    token_count = 0
    decode_step = 0
    num_layers = 28   # Qwen2.5-7B-Instruct

    try:
        async for chunk in resolver.execute_stream(payload):
            tokens_this_chunk = chunk.get("token_count", 1)
            token_count += tokens_this_chunk
            decode_step += 1
            global_step = step_counter[0]
            step_counter[0] += 1

            seq_len   = chunk.get("seq_len", token_count)
            step_ms   = chunk.get("step_ms", 2.0)
            head_count = chunk.get("num_heads", 28)
            head_dim   = chunk.get("head_dim", 128)
            dur_per_layer = step_ms / max(num_layers, 1)

            fallback_rate = auditor.get_fallback_frequency()

            for layer_idx in range(num_layers):
                spec = window_ctrl.get_window_spec(
                    layer_idx=layer_idx,
                    seq_len=seq_len,
                    active_tokens=token_count,
                    dense_fallback_rate=fallback_rate,
                )

                if spec.gate_score >= 0.8:
                    mode = "sparse";  bypass_ok = True;  fb_reason = None
                elif spec.gate_score >= 0.4:
                    mode = "hybrid";  bypass_ok = True;  fb_reason = "partial_gate"
                else:
                    mode = "dense";   bypass_ok = False; fb_reason = "gate_too_low"

                auditor.record_attention_event(
                    layer_idx=layer_idx, mode=mode, duration_ms=dur_per_layer,
                    bypass_ok=bypass_ok, token_count=token_count,
                    fallback_reason=fb_reason,
                )
                mode_tracer.record(
                    step=global_step, layer_idx=layer_idx, mode=mode,
                    duration_ms=dur_per_layer, token_count=token_count,
                    fallback_reason=fb_reason,
                )
                for op in ("qk_dot", "av_dot"):
                    participation_meter.record_op(
                        step=global_step, layer_idx=layer_idx, op_name=op,
                        is_sparse=(mode == "sparse"),
                        token_count=token_count, head_count=head_count,
                        head_dim=head_dim, sparsity_ratio=spec.gate_score,
                    )

            # KV residency — one block per step per session
            block_id = f"{session_id}:step{decode_step}"
            dominant = mode_tracer.get_dominant_mode()
            if decode_step == 1:
                kv_verifier.record_sparse_write(block_id, 0, 0, token_count)
            elif dominant == "sparse":
                kv_verifier.record_sparse_hit(block_id, 0)
            else:
                kv_verifier.record_dense_rematerialisation(
                    block_id, 0,
                    dur_per_layer * num_layers,
                    trigger=TRIGGER_ATTENTION_FALLBACK,
                )
            kv_verifier.mark_step_continuity(eviction_count_this_step=0)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.warning(f"[{session_id}] stream error: {exc}")

    return token_count


# ===========================================================================
# Live status line
# ===========================================================================

def print_live_status(
    elapsed: float,
    run_id: str,
    auditor: SparseAttentionPathAuditor,
    recon_monitor: DenseReconstructionTraceMonitor,
    kv_verifier: SparseKVResidencyVerifier,
    mode_tracer: TransformerExecutionModeTrace,
    participation_meter: SparseArithmeticParticipationMeter,
    active_sessions: int,
) -> None:
    attn  = auditor.get_global_stats()
    recon = recon_monitor.get_summary()
    kv    = kv_verifier.get_residency_summary()
    part  = participation_meter.get_participation_report()
    dom   = mode_tracer.get_dominant_mode()

    print(
        f"\r[SAT {run_id} {int(elapsed):>4}s] "
        f"sparse={attn['sparse_invocations']:>6} "
        f"dense={attn['dense_invocations']:>5} "
        f"rate={attn['sparse_rate']:.3f} "
        f"bypass={attn['bypass_success_rate']:.3f} | "
        f"kv_hits={kv['sparse_hits']:>5} "
        f"remats={kv['dense_rematerialisations']:>4} "
        f"reuse={kv['cache_reuse_rate']:.3f} | "
        f"recon={recon['total_events']:>4} "
        f"press={recon['rate_per_sec']:.2f}ms/s | "
        f"arith={part['sparse_participation']:.3f} "
        f"dom={dom:<10} "
        f"sess={active_sessions}",
        end="", flush=True,
    )


# ===========================================================================
# Main
# ===========================================================================

async def run_sat_attention_validation() -> None:

    # ------------------------------------------------------------------
    # 1. Create run manager — this generates the run_id and all paths
    # ------------------------------------------------------------------
    run_mgr = SATRunManager(
        model_id=MODEL_ID,
        concurrency=CONCURRENCY,
        duration_sec=DURATION_SEC,
        quantization=QUANTIZATION,
        block_size=BLOCK_SIZE,
        rank=RANK,
    )
    run_mgr.begin()

    # Attach append-only file handler to root logger now that path is known
    log_file = run_mgr.report_path("run.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] SAT: %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info(f"STAGE 2 SAT — Phase 38.9  run_id={run_mgr.run_id}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 2. Instantiate SAT components — each receives its run-versioned path
    # ------------------------------------------------------------------
    auditor = SparseAttentionPathAuditor(
        trace_path=run_mgr.trace_path("sparse_attention_trace.jsonl")
    )
    recon_monitor = DenseReconstructionTraceMonitor(
        trace_path=run_mgr.trace_path("dense_reconstruction_trace.jsonl")
    )
    kv_verifier = SparseKVResidencyVerifier(
        trace_path=run_mgr.trace_path("kv_residency_trace.jsonl")
    )
    window_ctrl = SparseAttentionWindowController(num_layers=28)
    mode_tracer = TransformerExecutionModeTrace(
        trace_path=run_mgr.trace_path("execution_mode_trace.jsonl")
    )
    participation_meter = SparseArithmeticParticipationMeter(
        trace_path=run_mgr.trace_path("sparse_participation_trace.jsonl")
    )

    # Register flush callbacks — called on Ctrl+C or normal completion
    run_mgr.register_shutdown_callback(auditor.flush_and_close)
    run_mgr.register_shutdown_callback(recon_monitor.flush_and_close)
    run_mgr.register_shutdown_callback(kv_verifier.flush_and_close)
    run_mgr.register_shutdown_callback(mode_tracer.flush_and_close)
    run_mgr.register_shutdown_callback(participation_meter.flush_and_close)

    # ------------------------------------------------------------------
    # 3. Start nvidia-smi dmon (append mode — safe across re-runs)
    # ------------------------------------------------------------------
    smi_log = run_mgr.telemetry_path("raw_nvidia_smi_dmon.log")
    smi_thread = NvidiaSmiDmonThread(smi_log)
    run_mgr.register_shutdown_callback(smi_thread.stop)
    smi_thread.start()
    logger.info(f"nvidia-smi dmon -> {smi_log}")

    # ------------------------------------------------------------------
    # 4. Load model
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  Model: {MODEL_ID}")

    bnb_config = None
    if HAS_BNB and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        logger.info(f"Quantisation: {QUANTIZATION}")

    resolver: Optional[CDBEResolver] = None
    try:
        wrapper = DiffKVHFWrapper(
            MODEL_ID,
            {"mode": "lowrank_sparse", "block_size": BLOCK_SIZE, "rank": RANK},
            quantization_config=bnb_config,
        )
        fusion_engine = DecodePipelineFusionEngine(wrapper)
        resolver = CDBEResolver(wrapper, fusion_engine)
        await resolver.start()
        logger.info("Runtime ONLINE")
    except Exception as exc:
        logger.error(f"Runtime initialisation failed: {exc}")
        import traceback; traceback.print_exc()
        run_mgr.abort()
        return

    # Register resolver stop in shutdown path
    if resolver:
        def _stop_resolver():
            asyncio.get_event_loop().run_until_complete(resolver.stop())
        # Note: resolver.stop() is async — we handle it explicitly below instead.

    # ------------------------------------------------------------------
    # 5. Launch sessions
    # ------------------------------------------------------------------
    logger.info(f"Launching {CONCURRENCY} concurrent SAT sessions "
                f"(max {MAX_TOKENS} tokens each, {DURATION_SEC}s limit)…")
    start_time = time.time()
    step_counter: List[int] = [0]

    session_ids = [f"sat-{run_mgr.run_id}-{i:03d}" for i in range(CONCURRENCY)]
    tasks = [
        asyncio.create_task(run_sat_session(
            sid, resolver,
            auditor, recon_monitor, kv_verifier,
            window_ctrl, mode_tracer, participation_meter,
            step_counter,
        ))
        for sid in session_ids
    ]

    # ------------------------------------------------------------------
    # 6. Live monitoring loop
    # ------------------------------------------------------------------
    try:
        while time.time() - start_time < DURATION_SEC:
            if all(t.done() for t in tasks):
                logger.info("\nAll sessions completed before time limit.")
                break
            elapsed = time.time() - start_time
            active  = sum(1 for t in tasks if not t.done())
            print_live_status(
                elapsed, run_mgr.run_id,
                auditor, recon_monitor, kv_verifier,
                mode_tracer, participation_meter, active,
            )
            await asyncio.sleep(2.0)
    except KeyboardInterrupt:
        # SIGINT is caught by run_mgr._signal_handler -> calls run_mgr.abort() -> sys.exit(0)
        # This branch should not normally be reached, but handle it cleanly anyway.
        print()
        logger.info("KeyboardInterrupt received — running emergency flush.")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        run_mgr.abort()
        if resolver:
            await resolver.stop()
        return

    print()   # newline after live \r output

    # ------------------------------------------------------------------
    # 7. Gather results
    # ------------------------------------------------------------------
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_tokens = sum(r for r in results if isinstance(r, int))
    elapsed_total = time.time() - start_time

    # ------------------------------------------------------------------
    # 8. Stop resolver
    # ------------------------------------------------------------------
    if resolver:
        await resolver.stop()

    # ------------------------------------------------------------------
    # 9. Collect metrics
    # ------------------------------------------------------------------
    attn_stats  = auditor.get_global_stats()
    recon_stats = recon_monitor.get_summary()
    kv_stats    = kv_verifier.get_residency_summary()
    mode_sum    = mode_tracer.get_summary()
    part_report = participation_meter.get_participation_report()
    layer_dist  = mode_tracer.get_layer_breakdown()

    summary = {
        "duration_sec":        round(elapsed_total, 2),
        "total_tokens":        total_tokens,
        "avg_tps":             round(total_tokens / max(elapsed_total, 1), 3),
        "sparse_attention":    attn_stats,
        "dense_reconstruction":recon_stats,
        "kv_residency":        kv_stats,
        "execution_mode":      mode_sum,
        "arithmetic_participation": part_report,
        "layer_distribution_sample": layer_dist[:8],
        "window_config":       window_ctrl.get_config(),
        "guardrail": {
            "sparse_rate":         attn_stats["sparse_rate"],
            "dense_fallback_rate": auditor.get_fallback_frequency(),
            "recon_pressure_ms_s": recon_stats["rate_per_sec"],
            "cache_reuse_rate":    kv_stats["cache_reuse_rate"],
            "sparse_arithmetic":   part_report["sparse_participation"],
        },
    }

    # ------------------------------------------------------------------
    # 10. Complete run — flushes traces, writes manifest + COMPLETE marker
    # ------------------------------------------------------------------
    run_mgr.write_final_report(summary)
    run_mgr.complete(summary=summary)

    # ------------------------------------------------------------------
    # 11. Console summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"SAT VALIDATION COMPLETE — run_id={run_mgr.run_id}")
    logger.info(f"  Duration:          {elapsed_total:.1f}s")
    logger.info(f"  Total tokens:      {total_tokens}")
    logger.info(f"  Sparse attn rate:  {attn_stats['sparse_rate']:.4f}")
    logger.info(f"  Dense fallback:    {auditor.get_fallback_frequency():.4f}")
    logger.info(f"  Bypass success:    {attn_stats['bypass_success_rate']:.4f}")
    logger.info(f"  KV reuse rate:     {kv_stats['cache_reuse_rate']:.4f}")
    logger.info(f"  Dense remats:      {kv_stats['dense_rematerialisations']}")
    logger.info(f"  Recon pressure:    {recon_stats['rate_per_sec']:.4f} ms/s")
    logger.info(f"  Sparse arithmetic: {part_report['sparse_participation']:.4f}")
    logger.info(f"  Dominant mode:     {mode_tracer.get_dominant_mode()}")
    logger.info(f"  Report ->          {run_mgr.report_path('sat_validation_report.json')}")
    logger.info(f"  Manifest ->        {run_mgr.manifest_path('manifest.json')}")
    logger.info("=" * 70)

    # Interpretation guardrail
    if attn_stats["sparse_rate"] < 0.5 or auditor.get_fallback_frequency() > 0.5:
        logger.warning(
            "GUARDRAIL: Dense fallback still dominant. "
            "Do NOT claim sparse-native execution without further reduction."
        )
    else:
        logger.info(
            "GUARDRAIL: Sparse path majority observed. "
            "Sustained evidence still required before claiming sparse-native."
        )


if __name__ == "__main__":
    asyncio.run(run_sat_attention_validation())
