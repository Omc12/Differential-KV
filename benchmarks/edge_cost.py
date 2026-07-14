"""Decode TPS + MLX peak-memory A/B for Relational Edge Capture (REC).
Same prompt/context both runs; only DIFFKV_RESIDUAL_EDGE_CAPTURE differs, so the
delta is REC's cost. Run once per --rec value in separate processes (env is read
at manager construction)."""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")


def mx_peak_gb():
    try:
        import mlx.core as mx
        for obj in (mx, getattr(mx, "metal", None)):
            fn = getattr(obj, "get_peak_memory", None) if obj else None
            if fn:
                v = fn()
                if v:
                    return v / 1e9
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True, choices=["0", "1"])
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--gen", type=int, default=96)
    args = ap.parse_args()

    os.environ["DIFFKV_COMPRESSED_DECODE"] = "1"
    os.environ["DIFFKV_FACTUAL_STORE"] = "0"
    os.environ["DIFFKV_RESIDUAL_EDGE_CAPTURE"] = args.rec

    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from transformers import AutoTokenizer
    import mlx.core as mx
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    with open(os.path.join(HERE, "random_features_paper.txt")) as f:
        paper = f.read()
    ftok = tok(paper, add_special_tokens=False)["input_ids"]
    reps = max(1, args.target // max(1, len(ftok)))
    body = (paper + "\n") * reps
    body = tok.decode(tok(body, add_special_tokens=False)["input_ids"][:args.target])
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": body + "\n\nSummarize the key relationship between the number of random features and test error."}],
        tokenize=False, add_generation_prompt=True)

    cfg = {"quantization": "int4", "rank": 16, "block_size": 256, "preset": "mid"}
    w = DiffKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=cfg)
    w.ensure_loaded()
    w.active_session = f"cost_{args.rec}"

    for fn in ("reset_peak_memory",):
        for obj in (mx, getattr(mx, "metal", None)):
            f2 = getattr(obj, fn, None) if obj else None
            if f2:
                try: f2()
                except Exception: pass

    # Warm prefill+decode once (exclude load), then timed run.
    t0 = time.time()
    _ = w.generate(prompt, max_new_tokens=args.gen, temperature=0.0, top_p=1.0,
                   repetition_penalty=1.1, query_text="summarize relationship")
    dt = time.time() - t0
    tps = args.gen / dt if dt > 0 else None
    peak = mx_peak_gb()
    print("__COST__ " + json.dumps(dict(rec=args.rec, target=args.target, gen=args.gen,
          total_s=round(dt, 2), approx_tps=round(tps, 2) if tps else None,
          mx_peak_gb=round(peak, 3) if peak else None)))
    try: w.close()
    except Exception: pass


if __name__ == "__main__":
    main()
