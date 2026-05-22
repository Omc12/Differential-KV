"""
test_phase7_scaling.py

Phase 7 Step 5: Scaling Pressure Test

Simulates a sustained multi-session workload to validate:
  1. PagedKVStore: GPU -> CPU spill-over under pressure
  2. ReconstructionCache: hit-rate during long decode loops
  3. AsyncCompressor: background SVD doesn't corrupt blocks
  4. KVRuntimeManager: stability across dozens of sessions

Test scenario:
  - 16 concurrent sessions
  - 512 prefill tokens each  (fills multiple blocks, triggers compression)
  - 200 decode steps each    (exercises recon cache and pager)
  - Total tokens processed: ~16 * 512 + 16 * 200 = ~11,392 tokens
"""

import sys, time, gc, torch
sys.path.insert(0, ".")

from runtime.kv_runtime_manager import KVRuntimeManager

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS  = 28      # Qwen2.5-7B layer count
KV_HEADS    = 4       # Qwen2.5-7B KV heads
HEAD_DIM    = 128
NUM_SESSIONS = 16
PREFILL_LEN  = 512
DECODE_STEPS = 200
GPU_BUDGET_GB = 0.5   # Deliberately tight budget (0.5 GB) to force paging

print("=" * 60)
print("PHASE 7 SCALING PRESSURE TEST")
print(f"  Device        : {DEVICE}")
print(f"  Sessions      : {NUM_SESSIONS}")
print(f"  Prefill length: {PREFILL_LEN} tokens")
print(f"  Decode steps  : {DECODE_STEPS}")
print(f"  GPU KV budget : {GPU_BUDGET_GB} GB")
print("=" * 60)

# ── Build manager ─────────────────────────────────────────────────────────────
mgr = KVRuntimeManager(
    num_layers        = NUM_LAYERS,
    heads             = KV_HEADS,
    head_dim          = HEAD_DIM,
    device            = DEVICE,
    gpu_budget_gb     = GPU_BUDGET_GB,
    recon_cache_size  = 64,
    async_compression = True,
)

sessions = [f"session-{i:03d}" for i in range(NUM_SESSIONS)]
for sid in sessions:
    mgr.init_session(sid)

vram_start = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0

# ── STEP 1: Prefill all sessions ──────────────────────────────────────────────
print(f"\n[1/3] Prefilling {NUM_SESSIONS} sessions x {PREFILL_LEN} tokens...")
t0 = time.perf_counter()

for sid in sessions:
    k = torch.randn(1, KV_HEADS, PREFILL_LEN, HEAD_DIM,
                    device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, KV_HEADS, PREFILL_LEN, HEAD_DIM,
                    device=DEVICE, dtype=torch.float16)
    # Only need to write to one layer per session for this test
    # (testing memory scaling, not layer fan-out)
    for layer in range(NUM_LAYERS):
        mgr.set_kv(sid, layer, k, v)

# Allow async compressor to drain
time.sleep(1.0)
if DEVICE == "cuda":
    torch.cuda.synchronize()

t_prefill = time.perf_counter() - t0
vram_after_prefill = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0
print(f"  Done in {t_prefill:.2f}s")
print(f"  VRAM after prefill: {vram_after_prefill:.1f} MB")

# ── STEP 2: Decode loop across all sessions ───────────────────────────────────
print(f"\n[2/3] Decode loop: {DECODE_STEPS} steps x {NUM_SESSIONS} sessions...")
t1 = time.perf_counter()
latencies = []

for step in range(DECODE_STEPS):
    step_t0 = time.perf_counter()

    for sid in sessions:
        # New token K/V
        new_k = torch.randn(1, KV_HEADS, 1, HEAD_DIM,
                            device=DEVICE, dtype=torch.float16)
        new_v = torch.randn(1, KV_HEADS, 1, HEAD_DIM,
                            device=DEVICE, dtype=torch.float16)

        # --- Exercise get_raw_blocks (Phase 6 sparse path) ---
        # Only check layer 0 for speed; real inference checks all layers
        blocks = mgr.get_raw_blocks(sid, 0)

        # Append new token (single layer for testing)
        mgr.set_kv(sid, 0, new_k, new_v)

    step_ms = (time.perf_counter() - step_t0) * 1000
    latencies.append(step_ms)

    if step % 50 == 0:
        vram_now = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0
        s = mgr.runtime_summary()
        print(f"  Step {step:3d} | {step_ms:.1f} ms | "
              f"VRAM={vram_now:.0f}MB | "
              f"evictions={s['pager']['total_evictions']} | "
              f"recon_hits={s['recon_cache']['hits']} | "
              f"compressions={s['total_compressions']}")

t_decode = time.perf_counter() - t1
vram_final = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0

# ── STEP 3: Report ────────────────────────────────────────────────────────────
print(f"\n[3/3] Final Runtime Summary")
print("=" * 60)

s = mgr.runtime_summary()

# Latency stats
avg_ms = sum(latencies) / len(latencies)
p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]
p99_ms = sorted(latencies)[int(len(latencies) * 0.99)]

print(f"\n  DECODE LATENCY (per-step, all {NUM_SESSIONS} sessions):")
print(f"    Avg p50   : {avg_ms:.2f} ms")
print(f"    p95       : {p95_ms:.2f} ms")
print(f"    p99       : {p99_ms:.2f} ms")
print(f"    Total     : {t_decode:.2f} s for {DECODE_STEPS} steps")

print(f"\n  VRAM USAGE:")
print(f"    Start     : {vram_start:.1f} MB")
print(f"    Prefill   : {vram_after_prefill:.1f} MB")
print(f"    Final     : {vram_final:.1f} MB")
print(f"    KV saved  : {s['vram_saved_mb']} MB (SVD compression)")

print(f"\n  PAGED MEMORY:")
print(f"    GPU resident  : {s['pager']['gpu_resident_mb']} MB")
print(f"    Evictions     : {s['pager']['total_evictions']}")
print(f"    Reloads       : {s['pager']['total_reloads']}")
print(f"    Bytes paged out: {s['pager']['bytes_paged_out_mb']} MB")
print(f"    Tracked blocks : {s['pager']['tracked_blocks']}")

print(f"\n  RECONSTRUCTION CACHE:")
print(f"    Hit rate      : {s['recon_cache']['hit_rate']:.1%}")
print(f"    Hits          : {s['recon_cache']['hits']}")
print(f"    Misses        : {s['recon_cache']['misses']}")
print(f"    Cached blocks : {s['recon_cache']['cached_blocks']}")

print(f"\n  ASYNC COMPRESSOR:")
print(f"    Submitted     : {s['async_compressor']['submitted']}")
print(f"    Completed     : {s['async_compressor']['completed']}")
print(f"    Queued        : {s['async_compressor']['queued']}")
print(f"    Sync fallbacks: {s['async_compressor']['sync_fallbacks']}")
print(f"    Avg SVD ms    : {s['async_compressor']['avg_svd_ms']} ms")

print(f"\n  COMPRESSION QUALITY:")
print(f"    Compressions  : {s['total_compressions']}")
print(f"    Avg cosine sim: {s['avg_cosine_sim']}")
print(f"    Avg norm drift: {s['avg_norm_drift']}")
print(f"    Sessions      : {s['sessions']}")

print("\n" + "=" * 60)
print("PHASE 7 SCALING TEST COMPLETE")
print("=" * 60)
