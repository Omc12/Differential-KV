"""Edge-fidelity A/B with INTERFERENCE (the real smear condition).

Four regimes share identical phrasing but a DIFFERENT directional relation, so the
low-energy relational verb (increases/reduces/unchanged/doubles) is exactly what
the SVD pool tends to smear — the decoder then reports a neighbouring regime's
direction. The registry is placed early and padded with real paper prose so it is
reached through compressed blocks.

Arms: dense | sparse_off (REC=0) | sparse_on (REC=1).
"""
import os, sys, json, argparse, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

# (regime, verb-phrase in registry, right-regex, wrong-regex[competitors])
REGIMES = [
    ("Zephyr",  "increases the reconstruction variance",
     r"increas", r"reduc|decreas|lower|unchang|same|doubl"),
    ("Marlowe", "reduces the reconstruction variance",
     r"reduc|decreas|lower", r"increas|unchang|same|doubl"),
    ("Corvid",  "leaves the reconstruction variance unchanged",
     r"unchang|same|constant|no change|does not", r"increas|reduc|decreas|doubl"),
    ("Pallas",  "doubles the reconstruction variance",
     r"doubl|two|2x|twice", r"reduc|decreas|unchang|same"),
]
REG_LINE = "In the {name} regime, raising the smoothing coefficient {verb}.\n"
Q = ("In the {name} regime, what happens to the reconstruction variance when the "
     "smoothing coefficient is raised? Answer in one short sentence.")


def load_filler():
    with open(os.path.join(HERE, "random_features_paper.txt")) as f:
        return f.read()


def build_prompt(tokenizer, target_tokens, depth=0.08):
    def templ(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    registry = ("Experimental regime notes (each regime behaves differently):\n" +
                "".join(REG_LINE.format(name=n, verb=v) for n, v, _, _ in REGIMES) + "\n")
    filler = load_filler()
    ftok = tokenizer(filler, add_special_tokens=False)["input_ids"]
    reps = max(1, target_tokens // max(1, len(ftok)))
    full = (filler + "\n") * reps
    fids = tokenizer(full, add_special_tokens=False)["input_ids"]
    ins = int(len(fids) * depth)
    body = tokenizer.decode(fids[:ins]) + "\n\n" + registry + "\n" + tokenizer.decode(fids[ins:])
    return body, templ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dense", "sparse_off", "sparse_on"])
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--gen", type=int, default=40)
    args = ap.parse_args()

    os.environ["DKV_COMPRESSED_DECODE"] = "0" if args.mode == "dense" else "1"
    os.environ["DKV_FACTUAL_STORE"] = "0"
    os.environ["DKV_RESIDUAL_EDGE_CAPTURE"] = "0" if args.mode == "sparse_off" else "1"

    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    body, templ = build_prompt(tok, args.target)
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256, "preset": "mid"}
    w = DKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=cfg)
    w.ensure_loaded()

    results = []
    n_ok = 0
    for i, (name, verb, right, wrong) in enumerate(REGIMES):
        sid = f"e2_{args.mode}_{i}"
        w.active_session = sid
        try: w.manager.clear_session(sid)
        except Exception: pass
        qtext = Q.format(name=name)
        prompt = templ(body + "\n\nQuestion: " + qtext)
        prompt_ids = tok.encode(prompt)
        _ = w.generate(prompt, max_new_tokens=args.gen, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.1, query_text=qtext)
        full_ids = w._session_token_ids.get(sid, [])
        gen_ids = full_ids[len(prompt_ids):] if len(full_ids) > len(prompt_ids) else full_ids
        out = tok.decode(gen_ids, skip_special_tokens=True).strip()
        has_right = re.search(right, out, re.I) is not None
        has_wrong = re.search(wrong, out, re.I) is not None
        ok = has_right and not has_wrong
        n_ok += int(ok)
        results.append(dict(regime=name, ok=ok, right=has_right, wrong=has_wrong, out=out[:120]))

    try: w.close()
    except Exception: pass
    print("__EDGE2__ " + json.dumps(dict(mode=args.mode, ctx=args.target,
                                         n_ok=n_ok, n=len(REGIMES), results=results)))


if __name__ == "__main__":
    main()
