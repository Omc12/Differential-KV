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

Open: why an unrotated pool hands SDPA more to chew on. The unrotated profile
also shows _fused_sparse_decode_kernel at 144 calls where the rotated one does
not, so the two configs may be taking different attention paths entirely rather
than the same path at different sizes -- that is the next thing to establish.

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
rows = sorted(ka, key=lambda e: -e.self_device_time_total)[:18]
total = sum(e.self_device_time_total for e in ka) / 1000.0
print(f"\nROT={os.environ['DKV_ROTATED_POOL']} total_self_cuda={total:.1f} ms "
      f"over {NEW} tokens ({total / NEW:.2f} ms/tok)", flush=True)
for e in rows:
    print(f"  {e.self_device_time_total / 1000.0:9.2f} ms  {e.count:6d}  {e.key[:70]}",
          flush=True)
