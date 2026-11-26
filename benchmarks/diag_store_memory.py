#!/usr/bin/env python3
"""Where does DKV's device memory go after a prefill? One point, one store.

Attributes torch.cuda.memory_allocated() after a 1-token generate to:
  weights     allocated before the first request
  pool.<f>    every tensor field of the native block pool
  blocks.<f>  every tensor attribute on the session's block objects
  workspace   tensors in the manager's decode workspace
  other       the remainder (anything held that none of the above owns)

Built to explain the granite codes-store spill (IMPROVEMENTS_LOG V5): run the
same context under DKV_STORE=codes and =lowrank and compare the rows. Short by
design: one prefill, no scoring.

  python benchmarks/diag_store_memory.py --model ibm-granite/granite-4.2-8b --ctx 32768
"""
import argparse
import gc
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)


def _tensor_bytes(obj, seen):
    import torch
    if isinstance(obj, torch.Tensor):
        if not obj.is_cuda:
            return 0
        key = obj.untyped_storage().data_ptr()
        if key in seen:
            return 0
        seen.add(key)
        return obj.untyped_storage().nbytes()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(x, seen) for x in obj.values())
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--preset", default="mid")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--json", default="")
    ap.add_argument("--trace-shape", default="",
                    help='print who holds one CUDA tensor of this shape, e.g. "(16, 1025, 8, 128)"')
    args = ap.parse_args()

    import torch
    from context_ladder import build_filler
    os.environ.setdefault("DKV_RSVD_SEED", "1234")
    cwd = os.getcwd()
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    from serving.decode_config import apply_best_decode_defaults
    apply_best_decode_defaults()
    from serving.hf_dkv_wrapper import DKVHFWrapper
    w = DKVHFWrapper(model_id=args.model,
                     config={"preset": args.preset, "quantization": args.quant or None})
    w.ensure_loaded()
    os.chdir(cwd)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()

    prompt = build_filler(w.tokenizer, args.ctx)
    sid = "diag-mem"
    w.active_session = sid
    torch.cuda.reset_peak_memory_stats()
    w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0,
               repetition_penalty=1.0)
    torch.cuda.synchronize()
    gc.collect()
    peak = torch.cuda.max_memory_allocated()
    alloc = torch.cuda.memory_allocated()

    seen = set()
    rows = defaultdict(int)
    m = w.manager
    pool = getattr(m, "native_pool", None)
    if pool is not None:
        for k, v in vars(pool).items():
            b = _tensor_bytes(v, seen)
            if b:
                rows[f"pool.{k}"] += b
        kbt = getattr(pool, "key_box_table", None)
        if kbt is not None:
            rows["pool.key_box_table"] += _tensor_bytes(
                [kbt.mn, kbt.mx, kbt.valid], seen)
    sm = getattr(m, "_streaming_mgr", None)
    n_blocks = defaultdict(int)
    if sm is not None:
        for _sid, layers in getattr(sm, "session_blocks", {}).items():
            for _li, blocks in layers.items():
                for b in blocks:
                    n_blocks[getattr(b, "state", "?")] += 1
                    for k, v in vars(b).items():
                        bb = _tensor_bytes(v, seen)
                        if bb:
                            rows[f"blocks.{k}"] += bb
    ws = getattr(m, "decode_workspace", None)
    if ws:
        for _wsid, sd in ws.items():
            if isinstance(sd, dict):
                for k, v in sd.items():
                    b = _tensor_bytes(v, seen)
                    if b:
                        rows[f"workspace.{k}"] += b
            else:
                rows["workspace"] += _tensor_bytes(sd, seen)
    for k, v in vars(m).items():
        if k in ("native_pool", "decode_workspace", "model", "_streaming_mgr"):
            continue
        b = _tensor_bytes(v, seen)
        if b:
            rows[f"manager.{k}"] += b

    # Module globals that hold CUDA tensors (caches nobody above owns).
    for modname, mod in list(sys.modules.items()):
        if not modname or not (modname.startswith(("runtime", "native_core", "serving"))):
            continue
        for k, v in list(vars(mod).items()):
            if isinstance(v, (dict, list, tuple)) or isinstance(v, torch.Tensor):
                b = _tensor_bytes(v, seen)
                if b:
                    rows[f"global.{modname}.{k}"] += b
    # Anything still unattributed: live CUDA tensors by shape.
    unowned = defaultdict(lambda: [0, 0])
    for o in gc.get_objects():
        try:
            if isinstance(o, torch.Tensor) and o.is_cuda:
                key = o.untyped_storage().data_ptr()
                if key in seen:
                    continue
                seen.add(key)
                sig = f"{tuple(o.shape)} {str(o.dtype).replace('torch.', '')}"
                unowned[sig][0] += o.untyped_storage().nbytes()
                unowned[sig][1] += 1
        except Exception:                                        # noqa: BLE001
            pass
    unowned_top = sorted(unowned.items(), key=lambda kv: -kv[1][0])[:12]

    # --trace-shape "(16, 1025, 8, 128)": who holds one tensor of that shape?
    trace = []
    if args.trace_shape:
        import inspect
        tgt = next((o for o in gc.get_objects() if isinstance(o, torch.Tensor)
                    and o.is_cuda and str(tuple(o.shape)) == args.trace_shape), None)
        frontier, depth = ([tgt] if tgt is not None else []), 0
        skip = {id(frontier)}
        while frontier and depth < 4:
            nxt = []
            for o in frontier:
                for r in gc.get_referrers(o):
                    if id(r) in skip or inspect.isframe(r) or r is frontier:
                        continue
                    skip.add(id(r))
                    desc = type(r).__name__
                    if isinstance(r, dict):
                        ks = [k for k, v in r.items() if v is o][:3]
                        desc += f" keys={ks}"
                    elif hasattr(r, "__qualname__"):
                        desc += f" {r.__qualname__}"
                    trace.append(f"{'  ' * depth}{desc}")
                    nxt.append(r)
            frontier, depth = nxt[:6], depth + 1

    accounted = sum(rows.values())
    out = {
        "model": args.model, "ctx": args.ctx,
        "store": __import__("runtime.native_block_pool", fromlist=["store_mode"]).store_mode(),
        "weights_gb": weights / 1e9, "alloc_gb": alloc / 1e9, "peak_gb": peak / 1e9,
        "kv_alloc_gb": (alloc - weights) / 1e9,
        "other_gb": (alloc - weights - accounted) / 1e9,
        "blocks_by_state": dict(n_blocks),
        "rows_gb": {k: v / 1e9 for k, v in sorted(rows.items(), key=lambda kv: -kv[1])},
        # Live tensors no owner above reaches (weights included -- compare
        # across stores, not to zero).
        "unowned_by_shape_gb": {k: [v[0] / 1e9, v[1]] for k, v in unowned_top},
        "trace": trace,
    }
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
