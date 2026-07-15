"""
profile_decode_step.py
Run on Lightning AI:
    cd /home/zeus/Differential-KV && python colab/profile_decode_step.py

Instruments a compressed decode step (after 4K prefill) and prints a
detailed per-phase timing breakdown to find exactly where time is going.
"""

import sys, os, time, gc
sys.path.insert(0, "/home/zeus/Differential-KV/ACTIVE_RUNTIME")
os.environ["DIFFKV_FACTUAL_STORE"]  = "0"
os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load DiffKV model
# ──────────────────────────────────────────────────────────────────────────────
from transformers import BitsAndBytesConfig
from serving.hf_diffkv_wrapper import DiffKVHFWrapper

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)
config = {
    "mode": "fp16", "quantization": "nf4",
    "block_size": 256, "rank": 16,
    "micro_block_size": 256, "preset": "mid", "serving_mode": "balanced"
}

print("Loading DiffKV model...")
t0 = time.perf_counter()
w = DiffKVHFWrapper(model_id=MODEL_ID, config=config,
                    torch_dtype=torch.float16, device=device,
                    quantization_config=quantization_config)
w.ensure_loaded()
tok, mgr, model = w.tokenizer, w.manager, w.model
print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Build 4K prompt and run prefill
# ──────────────────────────────────────────────────────────────────────────────
TARGET_LEN = 4096
WARMUP_TEXT = "The following is a long document. " * 300
ids = tok.encode(WARMUP_TEXT)[:TARGET_LEN]
print(f"Prompt tokens: {len(ids)}")

sid = "profile_session"
mgr.clear_session(sid)
mgr.init_session(sid, prefill_len=len(ids))
mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long, device=device))
model._diffkv_session_ids = [sid]

print("Running prefill...", flush=True)
CH = 128
out = None
t_pre = time.perf_counter()
with torch.inference_mode():
    for cs in range(0, len(ids), CH):
        ch = ids[cs:cs+CH]
        out = model(
            torch.tensor([ch], device=device),
            position_ids=torch.tensor([list(range(cs, cs+len(ch)))], device=device)
        )
        mgr.compress_deferred_prefill_blocks(sid)
print(f"Prefill done in {time.perf_counter()-t_pre:.2f}s\n")

if hasattr(mgr, "finalize_srl_index"):
    mgr.finalize_srl_index(sid, cached_len=0)
if hasattr(mgr, "_prefill_kv_capture"):
    mgr._prefill_kv_capture.pop(sid, None)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Warm up 3 decode steps (let JIT, CUDA graphs settle)
# ──────────────────────────────────────────────────────────────────────────────
last_logits = out.logits[0, -1].float()
static_in  = torch.zeros((1, 1), dtype=torch.long, device=device)
static_pos = torch.zeros((1, 1), dtype=torch.long, device=device)
cur = len(ids)

print("Warming up 3 decode steps...")
with torch.inference_mode():
    for _ in range(3):
        nid = int(torch.argmax(last_logits).item())
        static_in[0, 0] = nid
        static_pos[0, 0] = cur
        out = model(static_in, position_ids=static_pos)
        last_logits = out.logits[0, -1].float()
        cur += 1
torch.cuda.synchronize()
print("Warmup done.\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Per-step timing (20 steps, synchronized)
# ──────────────────────────────────────────────────────────────────────────────
N_STEPS = 20
step_times = []

print(f"Profiling {N_STEPS} decode steps (torch.cuda.synchronize before+after):")
with torch.inference_mode():
    for step in range(N_STEPS):
        pending_before = getattr(mgr, "_pending_cpu_blocks", 0)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        nid = int(torch.argmax(last_logits).item())
        t_argmax = time.perf_counter()

        static_in[0, 0] = nid
        static_pos[0, 0] = cur
        out = model(static_in, position_ids=static_pos)

        torch.cuda.synchronize()
        t_end = time.perf_counter()

        last_logits = out.logits[0, -1].float()
        cur += 1

        pending_after = getattr(mgr, "_pending_cpu_blocks", 0)
        total_ms  = (t_end    - t_start)  * 1000
        argmax_ms = (t_argmax - t_start)  * 1000
        model_ms  = (t_end    - t_argmax) * 1000
        step_times.append(total_ms)
        print(f"  Step {step+1:2d}: total={total_ms:7.1f}ms  "
              f"argmax={argmax_ms:.1f}ms  model={model_ms:.1f}ms  "
              f"pending={pending_before}")

avg_ms = sum(step_times) / len(step_times)
print(f"\nAvg: {avg_ms:.1f}ms/token  ({1000/avg_ms:.1f} TPS)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Phase-level instrumentation for ONE decode step
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Phase breakdown for a single decode step:")
print("=" * 60)

_orig_ingest = mgr.ingest_streaming
_ingest_ms = [0.0]; _ingest_n = [0]
def _timed_ingest(*a, **kw):
    t = time.perf_counter()
    r = _orig_ingest(*a, **kw)
    _ingest_ms[0] += (time.perf_counter()-t)*1000
    _ingest_n[0] += 1
    return r
mgr.ingest_streaming = _timed_ingest

_orig_blocks = mgr.get_cached_decode_blocks if hasattr(mgr, "get_cached_decode_blocks") else None
_blocks_ms = [0.0]; _blocks_n = [0]
if _orig_blocks:
    def _timed_blocks(*a, **kw):
        t = time.perf_counter()
        r = _orig_blocks(*a, **kw)
        _blocks_ms[0] += (time.perf_counter()-t)*1000
        _blocks_n[0] += 1
        return r
    mgr.get_cached_decode_blocks = _timed_blocks

with torch.inference_mode():
    nid = int(torch.argmax(last_logits).item())
    static_in[0, 0] = nid
    static_pos[0, 0] = cur
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(static_in, position_ids=static_pos)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000

# Print
print(f"  Total model forward:       {total_ms:.1f} ms")
print(f"  ingest_streaming:          {_ingest_ms[0]:.1f} ms  ({_ingest_n[0]} calls @ {_ingest_ms[0]/max(1,_ingest_n[0]):.3f} ms/call)")
if _orig_blocks:
    print(f"  get_cached_decode_blocks:  {_blocks_ms[0]:.1f} ms  ({_blocks_n[0]} calls @ {_blocks_ms[0]/max(1,_blocks_n[0]):.3f} ms/call)")
other = total_ms - _ingest_ms[0] - _blocks_ms[0]
print(f"  Remaining (GPU+other):     {other:.1f} ms")

mgr.ingest_streaming = _orig_ingest
if _orig_blocks:
    mgr.get_cached_decode_blocks = _orig_blocks

print(f"""
KEY: if 'model forward' >> dense baseline (~88ms), the bottleneck is inside
     the DiffKV decode path. If 'pending' stays > 0, background SVD is
     competing with decode. If 'ingest_streaming' is large, that's the hot path.
""")
