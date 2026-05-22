"""
Quick paging trigger test: 64 sessions x 1024 tokens, 100MB GPU budget.
Purpose: validate that PagedKVStore actually evicts blocks to CPU under real pressure.
"""
import sys, time, torch
sys.path.insert(0, ".")
from runtime.kv_runtime_manager import KVRuntimeManager

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_LAYERS    = 2
KV_HEADS      = 4
HEAD_DIM      = 128
PREFILL       = 1024
NUM_SESSIONS  = 64
GPU_BUDGET_GB = 0.1   # 100 MB — very tight to force evictions

print(f"Paging stress test: sessions={NUM_SESSIONS}, prefill={PREFILL}, budget={GPU_BUDGET_GB}GB")

mgr = KVRuntimeManager(
    num_layers       = NUM_LAYERS,
    heads            = KV_HEADS,
    head_dim         = HEAD_DIM,
    device           = DEVICE,
    gpu_budget_gb    = GPU_BUDGET_GB,
    recon_cache_size = 64,
    async_compression = False,   # sync for determinism
)

vram_before = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0

for i in range(NUM_SESSIONS):
    sid = f"sess-{i}"
    mgr.init_session(sid)
    k = torch.randn(1, KV_HEADS, PREFILL, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    v = torch.randn(1, KV_HEADS, PREFILL, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    for layer in range(NUM_LAYERS):
        mgr.set_kv(sid, layer, k, v)
    del k, v

# Give background pager a moment to run
time.sleep(1.0)

s = mgr.runtime_summary()
vram_after = torch.cuda.memory_allocated() / 1e6 if DEVICE == "cuda" else 0

print()
print(f"  VRAM before : {vram_before:.1f} MB")
print(f"  VRAM after  : {vram_after:.1f} MB")
print(f"  KV saved    : {s['vram_saved_mb']} MB  (SVD compression)")
print()
print(f"  PAGED MEMORY:")
print(f"    GPU resident  : {s['pager']['gpu_resident_mb']} MB")
print(f"    Evictions     : {s['pager']['total_evictions']}")
print(f"    Reloads       : {s['pager']['total_reloads']}")
print(f"    Bytes paged   : {s['pager']['bytes_paged_out_mb']} MB")
print(f"    Tracked blocks: {s['pager']['tracked_blocks']}")
print()
print(f"  Compressions  : {s['total_compressions']}")
print(f"  Avg cosine sim: {s['avg_cosine_sim']}")
print(f"  Sessions      : {s['sessions']}")

# Now simulate a decode step that reloads an evicted session
print()
print("  Touching first session (may trigger reload)...")
t0 = time.time()
blocks = mgr.get_raw_blocks("sess-0", 0)
reload_ms = (time.time() - t0) * 1000
s2 = mgr.runtime_summary()
print(f"  get_raw_blocks latency: {reload_ms:.2f} ms")
print(f"  Reloads after touch   : {s2['pager']['total_reloads']}")
