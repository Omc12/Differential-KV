"""Decode cost as a function of the routed block count K, in one process.

Each arm is a set of environment overrides read at call time (DKV_TOPK_BLOCKS,
DKV_FAST_ATTEND, ...). Arms are interleaved and their order rotates every
round, so clock drift lands on all of them; the statistic is the minimum
per-token time of a run (timing noise here is one-sided, see
bench_decode_paired.py).

    ARMS="k16:DKV_TOPK_BLOCKS=16;k64:DKV_TOPK_BLOCKS=64" REP=2800 \
        python colab/bench_k_ladder.py

Also prints the remat cache reuse of each run: a K whose materialised blocks do
not fit the remat budget is rebuilt every step, which is a cost cliff that has
nothing to do with attention itself.
"""
import io
import os
import re
import statistics
import sys
from contextlib import redirect_stdout

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, ROOT)
os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
from serving.decode_config import BEST_DECODE_DEFAULTS  # noqa: E402

for _k, _v in BEST_DECODE_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
os.environ["DKV_TIME_ATTN"] = "1"
from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
MODE = os.environ.get("WMODE", "fp16")
REP = int(os.environ.get("REP", "2800"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
NTOK = int(os.environ.get("NTOK", "24"))
_TOK = re.compile(r"total_token=([0-9.]+)ms")


def parse_arms(spec):
    arms = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, kv = part.partition(":")
        env = {}
        for item in kv.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                env[k.strip()] = v.strip()
        arms.append((name.strip(), env))
    return arms


def main():
    arms = parse_arms(os.environ.get(
        "ARMS", "k16:DKV_TOPK_BLOCKS=16;k32:DKV_TOPK_BLOCKS=32;all:DKV_TOPK_BLOCKS=0"))
    keys = sorted({k for _, e in arms for k in e})
    base = {k: os.environ.get(k) for k in keys}

    w = PyTorchDKVHFWrapper(model_id=MODEL, config={"mode": MODE}, device="cuda")
    w.ensure_loaded()
    tok = w.tokenizer
    ctx = "The archive records a long sequence of unremarkable events. " * REP
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": ctx + "\nWrite a detailed essay about archives."}],
        tokenize=False, add_generation_prompt=True)
    print(f"CTX {len(tok(prompt).input_ids)} model={MODEL} rounds={ROUNDS} ntok={NTOK}",
          flush=True)

    def apply(env):
        for k in keys:
            if k in env:
                os.environ[k] = env[k]
            elif base[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = base[k]

    def run(n):
        for _sid in list(getattr(w.manager, "decode_workspace", {}) or {}):
            try:
                w.manager.clear_session(_sid)
            except Exception:
                pass
        buf = io.StringIO()
        with redirect_stdout(buf):
            w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0)
        s = [float(m) for m in _TOK.findall(buf.getvalue())]
        reuse = ""
        for ws in (getattr(w.manager, "decode_workspace", {}) or {}).values():
            rc = ws.get("_remat_cache") if isinstance(ws, dict) else None
            if rc is not None:
                reuse = rc.stats()
        return (s[1:] or s), reuse

    apply(arms[0][1])
    run(NTOK)                                   # warm-up: clocks, autotune
    res = {name: [] for name, _ in arms}
    for r in range(ROUNDS):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for name, env in order:
            apply(env)
            torch.cuda.reset_peak_memory_stats()
            s, reuse = run(NTOK)
            res[name].append(min(s))
            print(f"  round {r} {name}: min={min(s):.2f} med={statistics.median(s):.2f} ms  "
                  f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB  {reuse}",
                  flush=True)
    print("RESULT")
    for name, _ in arms:
        v = res[name]
        print(f"  {name:>12}: {statistics.median(v):7.2f} ms/tok "
              f"({1000 / statistics.median(v):.2f} tok/s)  runs={['%.2f' % x for x in v]}",
              flush=True)


if __name__ == "__main__":
    main()
