"""Profile decode steps (host and device) after a long prefill, through generate().

Unlike profile_decode_step.py (which drives model() directly with its own
prompt and preset), this wraps the wrapper's real generate loop.

Wraps the model's forward and runs torch.profiler over decode calls
FIRST..FIRST+N-1 (1-token calls only; prefill chunks are skipped). Prints the top ops by self CPU time
and by self CUDA time, and the wall time of each profiled step, so a
host-bound step (CPU total >> CUDA total) is visible as such.

    REP=2800 STEPS=4 python colab/profile_decode_generate.py
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS  # noqa: E402

for _k, _v in BEST_DECODE_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
REP = int(os.environ.get("REP", "2800"))
FIRST = int(os.environ.get("FIRST", "4"))
STEPS = int(os.environ.get("STEPS", "4"))
ROWS = int(os.environ.get("ROWS", "25"))


def main():
    w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": os.environ.get("WMODE", "fp16")},
                            device="cuda")
    w.ensure_loaded()
    tok = w.tokenizer
    ctx = "The archive records a long sequence of unremarkable events. " * REP
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": ctx + "\nWrite a detailed essay about archives."}],
        tokenize=False, add_generation_prompt=True)
    model = w.model
    orig = model.forward
    state = {"n": 0, "prof": None, "walls": []}

    def fwd(*a, **k):
        ids = k.get("input_ids", a[0] if a else None)
        if ids is None or ids.shape[-1] != 1:
            return orig(*a, **k)              # prefill chunk: not counted
        i = state["n"]
        state["n"] += 1
        if i == FIRST:
            state["prof"] = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA], with_stack=os.environ.get("STACK") == "1", record_shapes=os.environ.get("SHAPES") == "1")
            state["prof"].__enter__()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **k)
        torch.cuda.synchronize()
        if FIRST <= i < FIRST + STEPS:
            state["walls"].append((time.perf_counter() - t0) * 1000)
        if i == FIRST + STEPS - 1:
            state["prof"].__exit__(None, None, None)
        return out

    model.forward = fwd
    w.generate(prompt=prompt, max_new_tokens=FIRST + STEPS + 1, temperature=0.0,
               top_p=1.0, repetition_penalty=1.0)
    p = state["prof"]
    print(f"\nWALL per profiled step (ms, profiler on): "
          f"{['%.1f' % x for x in state['walls']]}")
    ka = p.key_averages()
    cpu_tot = sum(e.self_cpu_time_total for e in ka) / 1000 / STEPS
    cuda_tot = sum(getattr(e, "self_device_time_total", 0) for e in ka) / 1000 / STEPS
    print(f"per step: self CPU {cpu_tot:.1f} ms, self CUDA {cuda_tot:.1f} ms")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=ROWS))
    print(ka.table(sort_by="self_device_time_total", row_limit=ROWS))
    if os.environ.get("STACK") == "1":
        # Which source lines issue the most device time.
        kb = p.key_averages(group_by_stack_n=6)
        rows = sorted(kb, key=lambda e: -getattr(e, "self_device_time_total", 0))
        for e in rows[:ROWS]:
            dev_ms = getattr(e, "self_device_time_total", 0) / 1000 / STEPS
            if dev_ms < 0.5:
                break
            print(f"\n{dev_ms:7.2f} ms/step  {e.key}  calls/step={e.count / STEPS:.0f}")
            for fr in (e.stack or [])[:6]:
                print("        " + fr)
    if os.environ.get("SHAPES") == "1":
        kb = p.key_averages(group_by_input_shape=True)
        rows = sorted(kb, key=lambda e: -getattr(e, "self_device_time_total", 0))
        for e in rows[:ROWS]:
            dev_ms = getattr(e, "self_device_time_total", 0) / 1000 / STEPS
            if dev_ms < 0.5:
                break
            print(f"{dev_ms:7.2f} ms/step  {e.key[:40]:40s} calls/step={e.count / STEPS:5.1f}  "
                  f"shapes={str(e.input_shapes)[:160]}")


if __name__ == "__main__":
    main()
