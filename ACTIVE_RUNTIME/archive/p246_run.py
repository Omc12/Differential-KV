import sys, gc, json
sys.path.insert(0, ".")
sys.path.insert(0, "./native_core")

import torch

DEVICE = "cuda"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def snap(label):
    torch.cuda.synchronize()
    s = torch.cuda.memory_stats(DEVICE)
    return {
        "label":    label,
        "alloc":    round(torch.cuda.memory_allocated(DEVICE) / 1e6, 2),
        "reserv":   round(torch.cuda.memory_reserved(DEVICE)  / 1e6, 2),
        "peak":     round(torch.cuda.max_memory_allocated(DEVICE) / 1e6, 2),
        "active":   round(s.get("active_bytes.all.current",  0) / 1e6, 2),
        "inactive": round(s.get("inactive_split_bytes.all.current", 0) / 1e6, 2),
    }

print("=" * 60)
print("PHASE 24.6 -- REAL VRAM RESIDENCY AUDIT")
print("=" * 60)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)
s0 = snap("pre-load")

from native_core.kv_runtime_manager import KVRuntimeManager
from runtime.dkv_attention import apply_dkv_attention_patch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, trust_remote_code=True
)
model.eval()

nlayers = model.config.num_hidden_layers
nheads  = model.config.num_attention_heads
hdim    = model.config.hidden_size // nheads
kvheads = getattr(model.config, "num_key_value_heads", nheads)
print(f"Model: layers={nlayers} heads={nheads} kv_heads={kvheads} head_dim={hdim}")

kvm = KVRuntimeManager(
    nlayers, kvheads, hdim,
    device=DEVICE, streaming_ingest=True,
    micro_block_size=16, async_compression=True,
)
apply_dkv_attention_patch(model, kvm)

s_loaded = snap("weights loaded")
weight_mb = round(s_loaded["alloc"] - s0["alloc"], 2)
print(f"Weight VRAM: {weight_mb} MB")

# -- Instrument get_kv --
recon_log = []
_orig_get_kv = kvm.get_kv

import types

def _instrumented_get_kv(self, sid, lidx):
    ba = torch.cuda.memory_allocated(DEVICE)
    r  = _orig_get_kv(sid, lidx)
    aa = torch.cuda.memory_allocated(DEVICE)
    if r[0] is not None:
        mb     = round((r[0].numel() + r[1].numel()) * 2 / 1e6, 4)
        dmb    = round((aa - ba) / 1e6, 4)
        seqlen = r[0].shape[2]
        recon_log.append({"layer": lidx, "seq": seqlen, "tensor_mb": mb, "delta_mb": dmb})
    return r

kvm.get_kv = types.MethodType(_instrumented_get_kv, kvm)

# -- Run sessions --
rows = []
for ptoks in [32, 128, 512]:
    recon_log.clear()
    sid = "sess_" + str(ptoks)
    kvm.init_session(sid)
    model._dkv_session_ids = [sid]

    # Build exactly ptoks tokens
    raw = "The quick brown fox jumps over the lazy dog. " * ((ptoks // 9) + 1)
    ids = tokenizer(raw, return_tensors="pt").input_ids[0, :ptoks].unsqueeze(0).to(DEVICE)
    pos = torch.arange(ptoks, device=DEVICE).unsqueeze(0)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    s_pre = snap("pre-prefill")
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=pos, use_cache=True)
    s_post = snap("post-prefill")

    prefill_recon_calls = len(recon_log)
    prefill_recon_mb    = round(sum(r["tensor_mb"] for r in recon_log), 3)
    prefill_recon_delta = round(sum(r["delta_mb"]  for r in recon_log), 3)
    max_seq_seen        = max((r["seq"] for r in recon_log), default=0)

    gc.collect()
    torch.cuda.empty_cache()
    s_gc = snap("post-gc")

    # 10 decode steps
    recon_log.clear()
    gen = [out.logits[0, -1].argmax().item()]
    for step in range(10):
        di = torch.tensor([[gen[-1]]], device=DEVICE)
        dp = torch.tensor([[ptoks + step]], device=DEVICE)
        with torch.no_grad():
            do = model(input_ids=di, position_ids=dp, use_cache=True)
        gen.append(do.logits[0, -1].argmax().item())
    s_dec = snap("post-decode")
    decode_recon_calls = len(recon_log)

    kvm.clear_session(sid)
    gc.collect()
    torch.cuda.empty_cache()
    s_tear = snap("post-teardown")

    kv_oh = round(s_post["peak"] - weight_mb, 2)
    row = {
        "ptoks":              ptoks,
        "peak_mb":            s_post["peak"],
        "kv_overhead_mb":     kv_oh,
        "post_gc_alloc":      s_gc["alloc"],
        "post_gc_reserv":     s_gc["reserv"],
        "post_gc_inactive":   s_gc["inactive"],
        "prefill_recon_calls": prefill_recon_calls,
        "prefill_recon_mb":   prefill_recon_mb,
        "prefill_recon_delta_mb": prefill_recon_delta,
        "max_reconstructed_seq": max_seq_seen,
        "decode_recon_calls": decode_recon_calls,
        "post_dec_alloc":     s_dec["alloc"],
        "post_tear_alloc":    s_tear["alloc"],
    }
    rows.append(row)
    print(f"[{ptoks:4d} tok]  peak={row['peak_mb']:7.1f} MB  kv_oh={kv_oh:6.1f} MB  "
          f"recon_calls={prefill_recon_calls:4d}  recon_mb={prefill_recon_mb:7.3f}  "
          f"dec_recon={decode_recon_calls}  gc_inactive={s_gc['inactive']:.1f}")

# Idle state
gc.collect()
torch.cuda.empty_cache()
s_idle = snap("idle")
alloc_oh = round(s_idle["reserv"] - s_idle["alloc"], 2)
print(f"IDLE  alloc={s_idle['alloc']:.1f} MB  reserv={s_idle['reserv']:.1f} MB  "
      f"allocator_cache={alloc_oh:.1f} MB")
print(f"streaming_active={kvm._streaming_mgr is not None}  micro_block={kvm.micro_block_size}")

# Save JSON
out_data = {
    "model":    MODEL_ID,
    "weight_mb": weight_mb,
    "rows":     rows,
    "idle": {"alloc": s_idle["alloc"], "reserv": s_idle["reserv"], "allocator_oh": alloc_oh},
}
with open("phase24_6_raw_results.json", "w") as f:
    json.dump(out_data, f, indent=2)
print("Saved phase24_6_raw_results.json")
