"""Where does the UNROTATED pool actually spend its extra time?

The unrotated pool buys exact dense parity on both retrieval and digit recall
and costs 43% of decode on a hybrid model, 137% on a dense-attention one. Three
plausible explanations were measured and all three are no-effect: recompute
frequency (remat interval 4 vs 16), the router's own rotation
(DKV_ROUTER_ROPE), and the residual exact-RoPE gather
(DKV_RESIDUAL_EXACT_ROPE). So stop reasoning and read the kernel timeline.

WHAT IT FOUND (Qwen3.5-2B, 32k, 24 steps), and why this file is kept:

    total self-CUDA   rotated 4401 ms      unrotated 4684 ms

    fmha_cutlassF     rotated  501.95 ms / 342 calls
                      UNROTAT  729.07 ms / 198 calls

+227 ms of a +283 ms delta -- 80% of the regression -- is inside SDPA, and NO
RoPE kernel appears in the delta at all. Unrotated makes FEWER attention calls
that each cost 2.5x more, so the operand is bigger rather than the arithmetic on
the way in. Anything aimed at making the rotation cheaper will measure zero.

RESOLVED. It is a DISPATCH difference, and printing the interesting kernels
unconditionally is what showed it -- a top-N list had hidden the fused kernel
below its cutoff in one arm, making a size difference and a dispatch difference
look alike:

    rotated     fmha 520.47 ms / 342 calls,  fused kernel ABSENT
    unrotated   fmha 751.69 ms / 198 calls,  fused kernel 62.61 ms / 144 calls

342 - 198 = 144 = 24 tokens x 6 attended layers, i.e. exactly the DKV layers
moving off the remat/SDPA path onto the Triton one. The cause is one line in
dkv_attention._remat_attend: it DECLINES outright when the pool is unrotated,
because that path's dense window is only rotated inside the sparse kernel. So
`ultra` does not pay for rotation -- it pays for losing the remat cache, which
remat_cache.py's own docstring measures at 6.95 -> 10.18 tok/s at 32k (~46%),
almost exactly the observed 43%.

The fix is specified at that decline site. Not applied here: it needs the dense
window's absolute token positions plumbed through assemble_dense_window_kv, and
a dense window rotated at guessed positions produces confident garbage rather
than an error.

Runs N decode steps under one setting and prints the top CUDA kernels by self
time. Run it twice (ROT=1 and ROT=0) and diff the two lists.

    ROT=1 python colab/profile_rotated_pool.py
    ROT=0 python colab/profile_rotated_pool.py

Env: ROT, MODEL, REP (context), NEW (tokens).
"""
import os
import sys

import torch

sys.path.insert(0, r"C:\Users\USER\Desktop\Differential KV\ACTIVE_RUNTIME")
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
os.environ["DKV_ROUTE_PROBE"] = "0"
from serving.decode_config import BEST_DECODE_DEFAULTS
for k, v in BEST_DECODE_DEFAULTS.items():
    os.environ.setdefault(k, v)
os.environ["DKV_ROTATED_POOL"] = os.environ.get("ROT", "1")
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper

REP = int(os.environ.get("REP", "2800"))
NEW = int(os.environ.get("NEW", "24"))

w = PyTorchDKVHFWrapper(model_id=os.environ.get("MODEL", "Qwen/Qwen3.5-2B"),
                        config={"mode": "fp16"}, device="cuda")
w.ensure_loaded()
tok = w.tokenizer
ctx = "The archive records a long sequence of unremarkable events. " * REP
prompt = tok.apply_chat_template(
    [{"role": "user", "content": ctx + "\nSummarise."}],
    tokenize=False, add_generation_prompt=True)

w.generate(prompt=prompt, max_new_tokens=4, temperature=0.0, top_p=1.0)  # warm

from torch.profiler import ProfilerActivity, profile
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    w.generate(prompt=prompt, max_new_tokens=NEW, temperature=0.0, top_p=1.0)
    torch.cuda.synchronize()

ka = prof.key_averages()
# Print the interesting kernels UNCONDITIONALLY, not just the top-N. A top-18
# list cut off at 40 ms in one arm and 65 ms in the other, which made a kernel
# that was merely BELOW THE CUTOFF look absent -- i.e. like a dispatch
# difference rather than a size difference. Those need telling apart.
for e in sorted(ka, key=lambda x: -x.self_device_time_total):
    if any(t in e.key for t in ("fused_sparse", "fmha", "flash", "sdpa")):
        print(f"  KEY {e.self_device_time_total / 1000.0:9.2f} ms  {e.count:6d}  "
              f"{e.key[:70]}", flush=True)
rows = sorted(ka, key=lambda e: -e.self_device_time_total)[:12]
total = sum(e.self_device_time_total for e in ka) / 1000.0
print(f"\nROT={os.environ['DKV_ROTATED_POOL']} total_self_cuda={total:.1f} ms "
      f"over {NEW} tokens ({total / NEW:.2f} ms/tok)", flush=True)
for e in rows:
    print(f"  {e.self_device_time_total / 1000.0:9.2f} ms  {e.count:6d}  {e.key[:70]}",
          flush=True)
