"""Are the router's key boxes in the keys' true post-RoPE frame?

Phase `dkv`: one RULER prompt through DKV (streaming forced by default) with
exact_key_boxes wrapped, so every recorded box is saved with its layer, anchor
and token count, alongside the same box computed WITHOUT rotation.
Phase `truth`: the same token ids through the plain model; the cache holds the
exact post-RoPE keys, from which the same per-block boxes are computed.
Phase `compare`: per layer, how well each DKV box matches the true one.

    python benchmarks/diag_key_box_frame.py dkv    --out X.pt [--len 16384]
    python benchmarks/diag_key_box_frame.py truth  --out X.pt
    python benchmarks/diag_key_box_frame.py compare --out X.pt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)


def _item(length, task, idx):
    from run_ruler_cuda import load_generated
    data = os.path.join(REPO, "paper", "results", "ruler_data", "Qwen3.5-4B")
    for it in load_generated(data):
        if it["length"] == length and it["task"] == task and it["idx"] == idx:
            return it
    raise SystemExit("item not found")


def phase_dkv(a):
    import torch
    os.environ.setdefault("DKV_STORE", "lowrank")
    os.environ.setdefault("DKV_PREFILL_SDPA", "1")
    os.environ.setdefault("DKV_STREAMING_COMPRESS", "auto")
    os.environ.setdefault("DKV_STREAM_AUTO_TOKENS", "4096")
    os.environ.setdefault("DKV_ROUTE_ADAPTIVE", "0")
    from run_ruler_cuda import build_ruler_prompt
    sys.path.insert(0, ACTIVE)
    cwd = os.getcwd()
    os.chdir(ACTIVE)
    from serving.decode_config import apply_best_decode_defaults
    apply_best_decode_defaults()
    from serving.hf_dkv_wrapper import DKVHFWrapper
    import native_core.srl.key_box as kb
    saved = []
    orig = kb.exact_key_boxes

    def wrapped(blocks_list, manager, chunk=16):
        out = orig(blocks_list, manager, chunk)
        if out is not None:
            anc = torch.stack([b.anchor_kv[0, 0] for b in blocks_list], 0)
            act = torch.cat([b.active_k for b in blocks_list], 0)
            raw = torch.cat([anc.unsqueeze(2).to(act.dtype), act], 2)
            for i, b in enumerate(blocks_list):
                saved.append(dict(layer=b.layer_idx, anchor=int(b.anchor_idx),
                                  n=int(raw.shape[2]),
                                  mn=out[0][i].float().cpu(), mx=out[1][i].float().cpu(),
                                  raw_mn=raw[i].amin(1).float().cpu(),
                                  raw_mx=raw[i].amax(1).float().cpu()))
        return out
    kb.exact_key_boxes = wrapped
    import native_core.compression.lowrank  # noqa: F401  (imports key_box lazily)
    w = DKVHFWrapper(model_id="Qwen/Qwen3.5-4B", config={"preset": "mid", "quantization": "nf4"})
    w.ensure_loaded()
    os.chdir(cwd)
    it = _item(a.len, a.task, a.idx)
    prompt = build_ruler_prompt(w.tokenizer, it)
    ids = w.tokenizer(prompt, return_tensors="pt").input_ids[0]
    w.active_session = "boxdiag"
    w.generate(prompt, max_new_tokens=2, temperature=0.0, top_p=1.0, repetition_penalty=1.0)
    print(f"recorded {len(saved)} block boxes over layers "
          f"{sorted(set(s['layer'] for s in saved))}")
    torch.save({"ids": ids, "boxes": saved, "rot_pool": _rot_pool()}, a.out)


def _rot_pool():
    try:
        from native_core.sparse_decode.triton_fused_decode import pool_stores_rotated_k
        return bool(pool_stores_rotated_k())
    except Exception as e:                                       # noqa: BLE001
        return repr(e)


def phase_truth(a):
    import torch
    from run_longbench_cuda import load_plain
    d = torch.load(a.out)
    tok, model = load_plain("Qwen/Qwen3.5-4B", "nf4", False)
    ids = d["ids"].unsqueeze(0).to(model.device)
    pkv = None
    with torch.no_grad():                    # chunked: a 16k SDPA mask is 16 GB
        for s in range(0, ids.shape[1], 2048):
            out = model(ids[:, s:s + 2048], past_key_values=pkv, use_cache=True,
                        logits_to_keep=1)
            pkv = out.past_key_values
    keys = {}
    layers = getattr(pkv, "layers", None)
    n_layers = len(layers) if layers is not None else len(pkv.key_cache)
    for i in range(n_layers):
        k = getattr(layers[i], "keys", None) if layers is not None else pkv.key_cache[i]
        if k is not None and getattr(k, "dim", lambda: 0)() == 4 and k.shape[2] == ids.shape[1]:
            keys[i] = k[0].float().cpu()                         # [H, T, D]
    print("full-attention layers in cache:", sorted(keys))
    d["truth_layers"] = keys
    torch.save(d, a.out)


def phase_compare(a):
    import torch
    d = torch.load(a.out)
    keys = d["truth_layers"]
    full = sorted(keys)
    print("pool stores rotated keys:", d["rot_pool"])
    by = {}
    for s in d["boxes"]:
        by.setdefault(s["layer"], []).append(s)
    dkv_layers = sorted(by)
    print("DKV layer ids:", dkv_layers, " model full-attn layers:", full)
    # DKV may index attention layers 0..7 or model layers; map by order if needed.
    mapping = {l: (l if l in keys else full[i]) for i, l in enumerate(dkv_layers)}
    for l in dkv_layers:
        K = keys[mapping[l]]
        errs, rerrs, rel = [], [], []
        for s in by[l]:
            a0, n = s["anchor"], s["n"]
            seg = K[:, a0:a0 + n]                                # [H, n, D]
            if seg.shape[1] != n:
                continue
            tmn, tmx = seg.amin(1), seg.amax(1)
            span = (tmx - tmn).abs().mean().item() + 1e-6
            errs.append(((s["mn"] - tmn).abs().mean() + (s["mx"] - tmx).abs().mean()).item() / 2 / span)
            rerrs.append(((s["raw_mn"] - tmn).abs().mean() + (s["raw_mx"] - tmx).abs().mean()).item() / 2 / span)
        print(f"layer {l}->{mapping[l]}: blocks {len(errs)}  box err/span {sum(errs)/max(1,len(errs)):.3f}"
              f"  unrotated err/span {sum(rerrs)/max(1,len(rerrs)):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dkv", "truth", "compare"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--len", type=int, default=16384)
    ap.add_argument("--task", default="niah_multiquery")
    ap.add_argument("--idx", type=int, default=0)
    a = ap.parse_args()
    {"dkv": phase_dkv, "truth": phase_truth, "compare": phase_compare}[a.phase](a)
