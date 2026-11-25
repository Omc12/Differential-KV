#!/usr/bin/env python3
"""Per-layer prefill diff: DKV vs plain transformers, at the last prompt position.

WHY. The V13 fidelity panel measured a FIRST-token KL of 0.9-2.6 on all three
gemma-4 models at 32k (top-1 still agreeing), and ad2b88d3 had already recorded
0.29-1.15 at 8k and accepted it. The first token comes from the prefill alone,
and on every other model DKV's prefill matches dense to ~0, so a gemma-only
first-token KL is a forward mismatch, not compression. This names the first
decoder layer whose output diverges.

Same prompt (niah_recall.build_prompt), same weights (nf4 / bf16 compute, the
bnb config copied from logit_fidelity._run_dense_true), same tokenization.

USAGE (one model per process)
    python benchmarks/diag_prefill_layer_diff.py --mode dense --model google/gemma-4-e2b-it
    python benchmarks/diag_prefill_layer_diff.py --mode dkv   --model google/gemma-4-e2b-it
    python benchmarks/diag_prefill_layer_diff.py --mode compare --model google/gemma-4-e2b-it
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTIVE = os.path.join(ROOT, "ACTIVE_RUNTIME")
BENCH = os.path.join(ROOT, "benchmarks")


def _out(args, mode):
    tag = args.model.replace("/", "_")
    return os.path.join(args.out_dir, f"prefill_diff_{tag}_{args.ctx}_{mode}.pt")


def _layer_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _hook_layers(layers, store, cur):
    hs = []
    for i, layer in enumerate(layers):
        def _h(_m, _inp, out, i=i):
            if cur["L"] > 1:                       # prefill call: keep the last
                store[i] = _layer_out(out)[0, -1].detach().float().cpu()
        hs.append(layer.register_forward_hook(_h))
    return hs


def run_dense(args):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DynamicCache)
    sys.path.insert(0, BENCH)
    sys.path.insert(0, ACTIVE)
    from niah_recall import build_prompt
    from runtime.dkv_attention import resolve_decoder_layers

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.sdpa_backend:
        # Restrict torch SDPA to one backend: isolates a fused-kernel defect
        # from a mask defect (the mask is the same under every backend).
        torch.backends.cuda.enable_flash_sdp(args.sdpa_backend == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(args.sdpa_backend == "efficient")
        torch.backends.cuda.enable_math_sdp(args.sdpa_backend == "math")
        torch.backends.cuda.enable_cudnn_sdp(args.sdpa_backend == "cudnn")
    kw = {"device_map": "cuda", "dtype": torch.bfloat16,
          **({"attn_implementation": args.attn} if args.attn else {}),
          "quantization_config": BitsAndBytesConfig(
              load_in_4bit=True, bnb_4bit_quant_type="nf4",
              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kw)
    model.eval()
    layers = resolve_decoder_layers(model)
    store, cur = {}, {"L": 2}
    hs = _hook_layers(layers, store, cur)
    ids = tok(build_prompt(tok, args.ctx, args.depth), return_tensors="pt").input_ids[0].tolist()
    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()
    with torch.inference_mode():
        for i in range(0, len(ids), args.chunk):
            ch = ids[i:i + args.chunk]
            cur["L"] = len(ch) if len(ch) > 1 else 2
            model(input_ids=torch.tensor([ch], device="cuda"),
                  position_ids=torch.arange(i, i + len(ch), device="cuda").unsqueeze(0),
                  past_key_values=cache, use_cache=True, logits_to_keep=1)
    for h in hs:
        h.remove()
    torch.save({"layers": store, "n_tok": len(ids), "patched": []}, _out(args, "dense"))
    print(f"[dense] {len(ids)} tok, {len(store)} layers saved", flush=True)


def run_dkv(args):
    os.environ.setdefault("DKV_COMPRESSED_DECODE", "1")
    os.environ.setdefault("DKV_POOL_BUDGET_GB", "2.0")
    os.environ.setdefault("DKV_ROUTE_PROBE", "0")
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    sys.path.insert(0, BENCH)
    import torch
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from niah_recall import build_prompt
    from runtime.dkv_attention import resolve_decoder_layers

    w = DKVHFWrapper(model_id=args.model, config={"quantization": "nf4", "preset": "mid"})
    w.ensure_loaded()
    if args.dkv_chunk:
        # One chunk = no prior-chunk history path; splits a defect between the
        # within-chunk attention and the attend-earlier-chunks path.
        w.manager.config.prefill_chunk_size = args.dkv_chunk
    tok, model = w.tokenizer, w.model
    layers = resolve_decoder_layers(model)
    for i in args.unpatch:
        # Restore the native attention forward: splits "DKV's attention is wrong"
        # from "the wrapper's chunked prefill around it is wrong".
        a = layers[i].self_attn
        a.forward = a._original_forward
        del a._original_forward
    patched = [i for i, l in enumerate(layers)
               if hasattr(getattr(l, "self_attn", None), "_original_forward")]
    store, cur, calls = {}, {"L": 0}, []

    def _pre(_m, _a, kw):
        ii = kw.get("input_ids")
        if ii is None and _a:
            ii = _a[0]
        cur["L"] = int(ii.shape[1]) if ii is not None and hasattr(ii, "shape") else 0
        calls.append(cur["L"])

    hp = model.register_forward_pre_hook(_pre, with_kwargs=True)
    hs = _hook_layers(layers, store, cur)
    prompt = build_prompt(tok, args.ctx, args.depth)
    w.active_session = "prefill-diff"
    w.generate(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0, repetition_penalty=1.0,
               query_text="What is the secret passcode? Repeat it exactly.")
    for h in hs:
        h.remove()
    hp.remove()
    n_tok = len(tok(prompt).input_ids)
    torch.save({"layers": store, "n_tok": n_tok, "patched": patched}, _out(args, "dkv"))
    print(f"[dkv] {n_tok} tok, {len(store)} layers saved, patched={patched}, "
          f"forward calls (input lengths) {calls}", flush=True)


def compare(args):
    import torch
    d = torch.load(_out(args, "dense"))
    k = torch.load(_out(args, "dkv"))
    print(f"tokens dense {d['n_tok']} dkv {k['n_tok']}  patched layers {k['patched']}")
    first = None
    print(f"{'layer':>5} {'rel err':>10} {'cosine':>9}  note")
    for i in sorted(d["layers"]):
        a, b = d["layers"][i], k["layers"].get(i)
        if b is None:
            print(f"{i:>5} {'missing':>10}")
            continue
        rel = float((a - b).norm() / a.norm().clamp_min(1e-12))
        cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
        note = "DKV" if i in k["patched"] else ""
        if first is None and rel > args.tol:
            first = i
            note += "  <-- first divergence"
        print(f"{i:>5} {rel:>10.5f} {cos:>9.5f}  {note}")
    print(f"first layer over rel err {args.tol}: {first}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dense", "dkv", "compare"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--dkv-chunk", type=int, default=0,
                    help="dkv only: override the prefill chunk size (0 = shipped)")
    ap.add_argument("--unpatch", type=int, nargs="*", default=[],
                    help="dkv only: layer indices to put back on the native forward")
    ap.add_argument("--attn", default="",
                    help="dense only: attn_implementation (eager / sdpa); empty = default")
    ap.add_argument("--sdpa-backend", default="",
                    choices=["", "math", "efficient", "flash", "cudnn"],
                    help="dense only: allow a single torch SDPA backend")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out-dir", default=os.environ.get("TEMP", "."))
    args = ap.parse_args()
    {"dense": run_dense, "dkv": run_dkv, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    main()
