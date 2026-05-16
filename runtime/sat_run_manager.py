"""
STAGE 2 — SAT: Run Manager
Phase 38.9 — Sparse Attention Transition

Single authority for:
  - generating a unique, timestamped run_id
  - constructing and creating all per-run directories
  - writing the run manifest (hardware, git, model, config)
  - coordinating graceful shutdown across all trace writers
  - writing a completion / abort marker

One SATRunManager instance is created at process start and passed
to every component that needs a file path.

Run layout (example run_id = 20260516_210700):
  traces/stage2/phase_38_9_sat/20260516_210700/
  telemetry/stage2/phase_38_9_sat/20260516_210700/
  reports/stage2/phase_38_9_sat/20260516_210700/
  benchmarks/stage2/phase_38_9_sat/20260516_210700/
  manifests/stage2/phase_38_9_sat/20260516_210700/
"""

import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("SATRunManager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE = "phase_38_9_sat"
STAGE = "stage2"

TRACE_NAMES = [
    "sparse_attention_trace.jsonl",
    "dense_reconstruction_trace.jsonl",
    "kv_residency_trace.jsonl",
    "execution_mode_trace.jsonl",
    "sparse_participation_trace.jsonl",
]

MARKER_RUNNING   = "RUN_IN_PROGRESS"
MARKER_COMPLETE  = "RUN_COMPLETE"
MARKER_ABORTED   = "RUN_ABORTED"


# ---------------------------------------------------------------------------
# SATRunManager
# ---------------------------------------------------------------------------

class SATRunManager:
    """
    Central lifecycle manager for a single SAT validation run.

    Usage:
        mgr = SATRunManager(model_id="...", concurrency=10, duration_sec=360)
        mgr.begin()             # creates dirs, writes manifest, sets SIGINT handler
        ...
        # pass mgr.trace_path("sparse_attention_trace.jsonl") to each component
        ...
        mgr.complete(summary)   # flushes everything, writes completion marker
    """

    def __init__(
        self,
        model_id: str,
        concurrency: int,
        duration_sec: int,
        quantization: str = "4bit_nf4",
        block_size: int = 64,
        rank: int = 16,
        run_id: Optional[str] = None,
        phase: str = PHASE,
        stage: str = STAGE,
    ):
        self.model_id = model_id
        self.concurrency = concurrency
        self.duration_sec = duration_sec
        self.quantization = quantization
        self.block_size = block_size
        self.rank = rank
        self.phase = phase
        self.stage = stage

        # Stable, sortable run identifier  (UTC)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Root paths (relative to project root)
        self._trace_root    = Path("traces")   / self.stage / self.phase / self.run_id
        self._telemetry_root= Path("telemetry")/ self.stage / self.phase / self.run_id
        self._report_root   = Path("reports")  / self.stage / self.phase / self.run_id
        self._benchmark_root= Path("benchmarks")/ self.stage / self.phase / self.run_id
        self._manifest_root = Path("manifests")/ self.stage / self.phase / self.run_id

        self._run_start_ts: Optional[float] = None
        self._shutdown_callbacks: List[Callable[[], None]] = []
        self._shutdown_called = False
        self._lock = threading.Lock()

        # Append-only run log (stdout + file)
        self._run_log_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Path helpers  (these never create directories — call begin() first)
    # ------------------------------------------------------------------

    def trace_path(self, filename: str) -> str:
        return str(self._trace_root / filename)

    def telemetry_path(self, filename: str) -> str:
        return str(self._telemetry_root / filename)

    def report_path(self, filename: str) -> str:
        return str(self._report_root / filename)

    def manifest_path(self, filename: str) -> str:
        return str(self._manifest_root / filename)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Create directories, write manifest, arm SIGINT handler."""
        self._run_start_ts = time.time()

        # Create all directories
        for d in [
            self._trace_root, self._telemetry_root,
            self._report_root, self._benchmark_root,
            self._manifest_root,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Set up append-only run log
        self._run_log_path = self._report_root / "run.log"

        self._log(f"SAT Run Manager — run_id={self.run_id}")
        self._log(f"Trace root:    {self._trace_root}")
        self._log(f"Telemetry root:{self._telemetry_root}")

        # Write initial manifest (hardware + config)
        self._write_manifest(status=MARKER_RUNNING, summary=None)

        # Write status marker
        self._write_marker(MARKER_RUNNING)

        # Arm Ctrl+C / SIGTERM handler
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._log("SIGINT/SIGTERM handlers armed.")
        logger.info(f"[RunManager] Run {self.run_id} begun — directories ready.")

    def register_shutdown_callback(self, cb: Callable[[], None]) -> None:
        """
        Register a zero-argument callable that will be called on graceful shutdown.
        Use this to flush trace buffers from each SAT component.
        """
        with self._lock:
            self._shutdown_callbacks.append(cb)

    def complete(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Called at normal end-of-run. Flushes everything and writes COMPLETE marker."""
        self._graceful_shutdown(aborted=False, summary=summary)

    def abort(self) -> None:
        """Called on Ctrl+C / SIGTERM. Flushes what it can and writes ABORTED marker."""
        self._graceful_shutdown(aborted=True, summary=None)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def write_final_report(self, report: Dict[str, Any]) -> None:
        path = self.report_path("sat_validation_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
        self._log(f"Validation report -> {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _signal_handler(self, signum, frame):
        print("\n[RunManager] Signal received — initiating graceful shutdown…",
              flush=True)
        self.abort()
        # Exit after cleanup so the event loop terminates
        sys.exit(0)

    def _graceful_shutdown(self, aborted: bool, summary: Optional[Dict]) -> None:
        with self._lock:
            if self._shutdown_called:
                return
            self._shutdown_called = True

        marker = MARKER_ABORTED if aborted else MARKER_COMPLETE
        status_str = "ABORTED" if aborted else "COMPLETE"

        self._log(f"Graceful shutdown initiated — status={status_str}")

        # Fire all registered flush callbacks
        for cb in self._shutdown_callbacks:
            try:
                cb()
            except Exception as exc:
                self._log(f"  [WARN] Shutdown callback raised: {exc}")

        # Write final manifest
        self._write_manifest(status=marker, summary=summary)

        # Write status marker file
        self._write_marker(marker)

        elapsed = round(time.time() - self._run_start_ts, 2) if self._run_start_ts else 0
        self._log(f"Shutdown complete — elapsed={elapsed}s")

    def _write_marker(self, marker: str) -> None:
        path = self._manifest_root / f"{marker}.txt"
        with open(path, "w") as f:
            f.write(f"{marker}\n")
            f.write(f"run_id={self.run_id}\n")
            f.write(f"ts={datetime.now(timezone.utc).isoformat()}\n")

    def _write_manifest(self, status: str, summary: Optional[Dict]) -> None:
        """Write (or overwrite) the run manifest with current hardware + config."""
        manifest: Dict[str, Any] = {
            "phase": "38.9-SAT",
            "run_id": self.run_id,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "python_version": sys.version,
            "git_commit": self._git_commit(),
            "model_id": self.model_id,
            "quantization": self.quantization,
            "block_size": self.block_size,
            "rank": self.rank,
            "concurrency": self.concurrency,
            "duration_sec": self.duration_sec,
            "gpu": self._gpu_info(),
            "cuda_version": self._cuda_version(),
            "paths": {
                "traces":    str(self._trace_root),
                "telemetry": str(self._telemetry_root),
                "reports":   str(self._report_root),
                "manifests": str(self._manifest_root),
            },
            "trace_files": TRACE_NAMES,
            "telemetry_files": ["raw_nvidia_smi_dmon.log"],
        }
        if summary:
            manifest["summary"] = summary

        path = self._manifest_root / "manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=4)

    def _log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{ts}] {msg}"
        logger.info(line)
        if self._run_log_path:
            with open(self._run_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ------------------------------------------------------------------
    # Environment probes
    # ------------------------------------------------------------------

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return "unavailable"

    @staticmethod
    def _gpu_info() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return (
                    f"{props.name} "
                    f"({props.total_memory // (1024**3)}GB, "
                    f"SM {props.major}.{props.minor})"
                )
        except Exception:
            pass
        return "unavailable"

    @staticmethod
    def _cuda_version() -> str:
        try:
            import torch
            return torch.version.cuda or "unavailable"
        except Exception:
            return "unavailable"
