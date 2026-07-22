"""
native_core/diagnostics.py

Validation and profiling utilities for the DKV optimization pipeline.
All functions are safe to call in production (no side effects unless explicitly noted).

Usage:
    from native_core.diagnostics import log_vram, log_block_states, TpsCounter

    log_vram("post_prefill_compression")
    log_block_states(kv_manager, session_id)

    tps = TpsCounter()
    for tok in decode_loop():
        tps.tick()
        if tps.steps % 10 == 0:
            print(f"[TPS] {tps.rate:.1f}")
"""

import time
import torch
from collections import Counter
from typing import Optional


# ── VRAM audit ───────────────────────────────────────────────────────────────

def log_vram(label: str = "", device=None) -> dict:
    """
    Print and return current VRAM usage.

    Works on CUDA and MPS.
    On CPU returns zeros (no VRAM).

    Returns:
        dict with keys: label, allocated_gb, reserved_gb, free_gb (CUDA only)
    """
    result = {"label": label, "allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": None}

    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        alloc   = torch.cuda.memory_allocated(device)  / 1024**3
        reserv  = torch.cuda.memory_reserved(device)   / 1024**3
        total   = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free    = total - alloc
        result.update({"allocated_gb": alloc, "reserved_gb": reserv, "free_gb": free})
        print(f"[VRAM] {label}: {alloc:.2f} GB alloc, {reserv:.2f} GB reserved, "
              f"{free:.2f} GB free / {total:.2f} GB total")
    elif (hasattr(torch, "mps") and
          hasattr(torch.backends, "mps") and
          torch.backends.mps.is_available()):
        # MPS: unified memory — use driver-level stats if available
        try:
            alloc = torch.mps.current_allocated_memory() / 1024**3
            result["allocated_gb"] = alloc
            print(f"[VRAM] {label}: {alloc:.2f} GB allocated (MPS unified memory)")
        except Exception:
            print(f"[VRAM] {label}: MPS memory stats unavailable")
    else:
        print(f"[VRAM] {label}: CPU — no VRAM stats")

    return result


# ── Block state audit ─────────────────────────────────────────────────────────

def log_block_states(kv_manager, session_id: str) -> dict:
    """
    Print the count of KV blocks in each state for a given session.

    Expected output at decode start:
        [Blocks] {'COMPRESSED': 781}   ← all blocks ready
    Bad output (stale compression):
        [Blocks] {'COMPRESSED': 760, 'ACCUMULATING': 21}

    Returns: Counter dict of state → count.
    """
    states: Counter = Counter()

    mgr = kv_manager
    streaming_mgr = getattr(mgr, "_streaming_mgr", None)

    if streaming_mgr is not None and session_id in streaming_mgr.session_blocks:
        for layer_idx, blocks in streaming_mgr.session_blocks[session_id].items():
            for b in blocks:
                state = getattr(b, "state", "UNKNOWN")
                states[state] += 1
    elif hasattr(mgr, "session_blocks") and session_id in mgr.session_blocks:
        for layer_idx, blocks in mgr.session_blocks[session_id].items():
            for b in blocks:
                state = getattr(b, "state", "UNKNOWN")
                states[state] += 1

    pending = getattr(mgr, "_pending_cpu_blocks", "N/A")
    print(f"[Blocks] session={session_id}: {dict(states)}  "
          f"(pending_cpu={pending})")
    return dict(states)


# ── TPS measurement ───────────────────────────────────────────────────────────

class TpsCounter:
    """
    Lightweight tokens-per-second counter for the decode loop.

    Usage:
        counter = TpsCounter()
        for _ in range(max_tokens):
            # ... decode step ...
            counter.tick()
            if counter.steps % 10 == 0:
                print(f"[TPS] {counter.rate:.1f}")
    """
    def __init__(self):
        self._t0   = time.perf_counter()
        self.steps = 0

    def tick(self):
        self.steps += 1

    @property
    def rate(self) -> float:
        elapsed = time.perf_counter() - self._t0
        if elapsed < 1e-9:
            return 0.0
        return self.steps / elapsed

    def reset(self):
        self._t0   = time.perf_counter()
        self.steps = 0


# ── Pool state audit ──────────────────────────────────────────────────────────

def log_pool_stats(pool, label: str = "") -> dict:
    """
    Print pool allocation stats. Useful to confirm pool isn't over-allocated.

    Returns: dict with keys: total_blocks, used_blocks, free_blocks, vram_gb
    """
    total = getattr(pool, "current_blocks", 0)
    ref_counts = getattr(pool, "_ref_counts", [])
    used  = sum(1 for r in ref_counts if r > 0)
    free  = total - used

    # Estimate VRAM: U(int8) + U_scale(fp16) + V_KV(fp16) + anchors(fp16) + scales + seq_lens
    u_shape  = getattr(pool, "U", None)
    vkv_shape = getattr(pool, "V_KV", None)
    anc_shape = getattr(pool, "anchors_KV", None)
    vram_bytes = 0
    if u_shape is not None:
        vram_bytes += u_shape.element_size() * u_shape.nelement()
    if vkv_shape is not None:
        vram_bytes += vkv_shape.element_size() * vkv_shape.nelement()
    if anc_shape is not None:
        vram_bytes += anc_shape.element_size() * anc_shape.nelement()
    vram_gb = vram_bytes / 1024**3

    print(f"[Pool] {label}: {used}/{total} blocks used "
          f"({free} free), {vram_gb:.3f} GB pool VRAM")
    return {"total_blocks": total, "used_blocks": used,
            "free_blocks": free, "vram_gb": vram_gb}


# ── Kernel sync detector ──────────────────────────────────────────────────────

def enable_cuda_sync_debug():
    """
    Enable PyTorch's CUDA sync debug mode.
    Any unexpected CPU-GPU synchronization will raise a warning.
    Call once at startup for profiling sessions.
    Only available on CUDA.
    """
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode(1)
        print("[DKV Diagnostics] CUDA sync debug mode ENABLED. "
              "Unexpected sync points will print warnings.")
    else:
        print("[DKV Diagnostics] CUDA sync debug mode: N/A (no CUDA).")


def disable_cuda_sync_debug():
    """Disable CUDA sync debug mode (re-enable after profiling)."""
    if torch.cuda.is_available():
        torch.cuda.set_sync_debug_mode(0)
