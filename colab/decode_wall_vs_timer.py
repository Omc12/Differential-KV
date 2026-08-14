"""Wall-clock decode vs the per-token attention timer.

bench_matrix reports tok/s from DKV_TIME_ATTN, which measures the attention path.
Wall-clock generate() is ~2x dense at 16k while that timer says DKV and dense are
close, so the difference is time the timer does not see: sampling, the Python
generate loop, cache bookkeeping, everything between attention calls.
"""
import io, os, re, sys, time
from contextlib import redirect_stdout
import torch
ROOT = r"C:\Users\USER\Desktop\Differential KV"
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME")); sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS
for k, v in BEST_DECODE_DEFAULTS.items(): os.environ.setdefault(k, v)
os.environ["DKV_TIME_ATTN"] = "1"
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
REP = int(os.environ.get("REP", "1400"))
NEW = int(os.environ.get("NEW", "120"))
_TOK = re.compile(r"total_token=([0-9.]+)ms")
w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": "fp16",
        "preset": os.environ.get("PRESET", "mid")}, device="cuda")
w.ensure_loaded()
tok = w.tokenizer
ctx = "The archive records a long sequence of unremarkable events. " * REP
prompt = tok.apply_chat_template(
    [{"role": "user", "content": ctx + "\nWrite a long detailed essay about archives."}],
    tokenize=False, add_generation_prompt=True)
print(f"CTX {len(tok(prompt).input_ids)}", flush=True)

def run(n):
    buf = io.StringIO()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with redirect_stdout(buf):
        w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0,
                   repetition_penalty=1.0)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, [float(m) for m in _TOK.findall(buf.getvalue())]

run(4)                                    # warm
t1, s1 = run(1)                           # prefill + 1
tN, sN = run(NEW)
decode_wall = tN - t1                     # prefill cancels
steady = sN[1:]
timer_s = sum(steady) / 1000.0
n = len(steady)
print(f"\nDECODE {n} tokens after prefill")
print(f"  wall                {decode_wall:7.2f} s   ({n / decode_wall:5.2f} tok/s)")
print(f"  attention timer sum {timer_s:7.2f} s   ({n / timer_s:5.2f} tok/s)")
print(f"  UNTIMED overhead    {decode_wall - timer_s:7.2f} s   "
      f"({100.0 * (decode_wall - timer_s) / decode_wall:4.1f}% of wall, "
      f"{1000.0 * (decode_wall - timer_s) / n:5.1f} ms/token)")
