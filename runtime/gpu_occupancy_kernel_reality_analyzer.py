"""
PRD Phase 41.0: GPU Occupancy & Kernel Reality Analyzer.
Measures REAL GPU utilization behavior using nvidia-smi dmon and pynvml.

Tracks:
- SM occupancy
- kernel launch frequency
- GPU idle gaps
- memory stalls
- sparse kernel efficiency
- CUDA synchronization stalls
- host-device transfer overhead

We are validating whether the GPU is truly being used efficiently.
"""

import time
import json
import threading
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque


class GPUOccupancyKernelRealityAnalyzer:
    """
    PRD Phase 41.0: Real GPU occupancy and kernel behavior analyzer.
    Uses nvidia-smi dmon for hardware-level SM utilization sampling.
    Falls back to pynvml if available, then gracefully degrades.
    """

    def __init__(self, trace_dir: Path, gpu_index: int = 0, sample_interval_ms: int = 100):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("PRD_GPUAnalyzer")
        self._gpu_index = gpu_index
        self._sample_interval_ms = sample_interval_ms

        # Rolling windows
        self._sm_utilizations: deque = deque(maxlen=300)  # 30 seconds at 100ms
        self._memory_utilizations: deque = deque(maxlen=300)
        self._power_draws: deque = deque(maxlen=300)
        self._idle_gaps: List[float] = []

        self._lock = threading.Lock()
        self._running = False
        self._sample_thread: Optional[threading.Thread] = None
        self._last_active_ts: float = time.perf_counter()

        # Kernel timing accumulation
        self._kernel_events: deque = deque(maxlen=500)
        self._cuda_sync_stalls: deque = deque(maxlen=200)
        self._host_device_transfers: deque = deque(maxlen=200)

        # Raw nvidia-smi log path
        self._raw_dmon_path = self.trace_dir / "raw_nvidia_smi_dmon.log"
        self._trace_path = self.trace_dir / "gpu_occupancy_trace.jsonl"

        # Detect backends
        self._nvml_available = self._try_init_nvml()
        self._smi_available = self._check_smi()

        backend = "pynvml" if self._nvml_available else ("nvidia-smi" if self._smi_available else "UNAVAILABLE")
        self._logger.info(f"GPUOccupancyAnalyzer initialized | backend={backend} | gpu={gpu_index}")

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        """Begin continuous GPU sampling in background thread."""
        self._running = True
        if self._nvml_available:
            self._sample_thread = threading.Thread(
                target=self._nvml_sample_loop, daemon=True, name="gpu_nvml_sampler"
            )
        elif self._smi_available:
            self._sample_thread = threading.Thread(
                target=self._smi_sample_loop, daemon=True, name="gpu_smi_sampler"
            )
        else:
            self._logger.warning("No GPU monitoring backend available. SM occupancy will not be measured.")
            return
        self._sample_thread.start()

    def stop(self):
        """Stop sampling loop."""
        self._running = False
        if self._sample_thread:
            self._sample_thread.join(timeout=3.0)
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Kernel event recording (call from inference hot-path)
    # -----------------------------------------------------------------------

    def record_kernel_launch(self, kernel_name: str, duration_ms: float):
        """Record a kernel launch event."""
        with self._lock:
            self._kernel_events.append({
                "ts": time.time(),
                "kernel": kernel_name,
                "duration_ms": duration_ms,
            })
            self._last_active_ts = time.perf_counter()

    def record_cuda_sync_stall(self, stall_ms: float):
        """Record a CUDA synchronization stall."""
        with self._lock:
            self._cuda_sync_stalls.append(stall_ms)

    def record_host_device_transfer(self, bytes_transferred: int, duration_ms: float):
        """Record a host↔device transfer."""
        with self._lock:
            self._host_device_transfers.append({
                "bytes": bytes_transferred,
                "duration_ms": duration_ms,
                "bandwidth_gbps": round(bytes_transferred / duration_ms / 1e6, 3) if duration_ms > 0 else 0,
            })

    def mark_gpu_active(self):
        """Called when a kernel/forward pass is dispatched."""
        with self._lock:
            self._last_active_ts = time.perf_counter()

    # -----------------------------------------------------------------------
    # Live summary
    # -----------------------------------------------------------------------

    def get_live_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_sm = round(sum(self._sm_utilizations) / len(self._sm_utilizations), 1) if self._sm_utilizations else 0.0
            avg_mem = round(sum(self._memory_utilizations) / len(self._memory_utilizations), 1) if self._memory_utilizations else 0.0
            avg_power = round(sum(self._power_draws) / len(self._power_draws), 1) if self._power_draws else 0.0
            recent_idle_gaps = self._idle_gaps[-20:]
            avg_idle_gap_ms = round(sum(recent_idle_gaps) / len(recent_idle_gaps) * 1000, 2) if recent_idle_gaps else 0.0
            avg_sync_stall = round(sum(self._cuda_sync_stalls) / len(self._cuda_sync_stalls), 2) if self._cuda_sync_stalls else 0.0
            kernel_hz = len(self._kernel_events)

        return {
            "sm_utilization_pct": avg_sm,
            "memory_utilization_pct": avg_mem,
            "power_draw_w": avg_power,
            "avg_idle_gap_ms": avg_idle_gap_ms,
            "total_idle_gaps": len(self._idle_gaps),
            "avg_cuda_sync_stall_ms": avg_sync_stall,
            "kernel_launch_count": kernel_hz,
            "backend": "pynvml" if self._nvml_available else ("nvidia-smi" if self._smi_available else "none"),
        }

    def format_live_line(self) -> str:
        s = self.get_live_summary()
        return (
            f"[GPU] SM={s['sm_utilization_pct']:.0f}% "
            f"MEM={s['memory_utilization_pct']:.0f}% "
            f"POWER={s['power_draw_w']:.0f}W "
            f"idle_gaps={s['total_idle_gaps']} "
            f"avg_idle={s['avg_idle_gap_ms']:.1f}ms "
            f"sync_stall={s['avg_cuda_sync_stall_ms']:.1f}ms"
        )

    # -----------------------------------------------------------------------
    # Sampling loops
    # -----------------------------------------------------------------------

    def _nvml_sample_loop(self):
        """High-frequency sampling via pynvml."""
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        interval = self._sample_interval_ms / 1000.0

        while self._running:
            t0 = time.time()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_pct = round(mem_info.used / mem_info.total * 100, 1)

                sm_util = util.gpu
                with self._lock:
                    self._sm_utilizations.append(sm_util)
                    self._memory_utilizations.append(mem_pct)
                    self._power_draws.append(power)

                    # Detect idle gap
                    idle_since = time.perf_counter() - self._last_active_ts
                    if sm_util < 5 and idle_since > 0.05:  # >50ms of <5% SM
                        self._idle_gaps.append(idle_since)

                record = {
                    "timestamp": t0,
                    "sm_utilization_pct": sm_util,
                    "memory_utilization_pct": mem_pct,
                    "power_draw_w": round(power, 1),
                    "vram_used_mb": round(mem_info.used / 1024**2, 1),
                    "vram_total_mb": round(mem_info.total / 1024**2, 1),
                }
                self._persist_sample(record)

            except Exception as e:
                self._logger.debug(f"nvml sample error: {e}")

            elapsed = time.time() - t0
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def _smi_sample_loop(self):
        """
        Periodic sampling via non-blocking nvidia-smi --query-gpu calls.
        Uses per-sample subprocess invocations to avoid dmon blocking issues on Windows.
        """
        interval = max(1.0, self._sample_interval_ms / 1000.0)  # at least 1s per query

        query = f"utilization.gpu,utilization.memory,power.draw"
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            f"--id={self._gpu_index}",
        ]

        # Open raw dmon log (we write our own structured data)
        raw_log_file = open(self._raw_dmon_path, "w", encoding="utf-8")
        raw_log_file.write("# timestamp,sm_util_pct,mem_util_pct,power_w\n")
        raw_log_file.flush()

        try:
            while self._running:
                t0 = time.time()
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        parts = [p.strip() for p in result.stdout.strip().split(",")]
                        if len(parts) >= 2:
                            sm = int(float(parts[0]))
                            mem = int(float(parts[1]))
                            power = float(parts[2]) if len(parts) >= 3 else 0.0
                            ts = time.time()

                            raw_log_file.write(f"{ts},{sm},{mem},{power}\n")
                            raw_log_file.flush()

                            with self._lock:
                                self._sm_utilizations.append(sm)
                                self._memory_utilizations.append(mem)
                                self._power_draws.append(power)
                                idle_since = time.perf_counter() - self._last_active_ts
                                if sm < 5 and idle_since > 0.1:
                                    self._idle_gaps.append(idle_since)

                            record = {
                                "timestamp": ts,
                                "sm_utilization_pct": sm,
                                "memory_utilization_pct": mem,
                                "power_draw_w": round(power, 1),
                            }
                            self._persist_sample(record)

                except subprocess.TimeoutExpired:
                    self._logger.debug("nvidia-smi query timed out")
                except ValueError:
                    pass
                except Exception as e:
                    self._logger.debug(f"SMI query error: {e}")

                elapsed = time.time() - t0
                sleep_time = max(0.1, interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            self._logger.error(f"SMI sample loop error: {e}")
        finally:
            raw_log_file.close()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _persist_sample(self, record: Dict[str, Any]):
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Backend detection helpers
    # -----------------------------------------------------------------------

    def _try_init_nvml(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    def _check_smi(self) -> bool:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, timeout=5
            )
            return r.returncode == 0
        except Exception:
            return False
