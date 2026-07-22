"""Context-Aware Decoding A/B in the REAL DKV compressed setting.
Prior-contradicting relational claims buried in real paper filler; compares
DKV_CAD_ALPHA off vs on. Forced-choice 2-word answers (no scorer contamination).
"""
import os, sys, argparse, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

CLAIMS = [
    ("In the Zephyr architecture, increasing the network depth strictly decreases final accuracy.",
     "In Zephyr, increasing depth makes final accuracy go up or down?", "down", "up",
     "Reply with exactly one word: up or down."),
    ("In the Marlowe system, adding more training data reduces generalization.",
     "In Marlowe, does adding more training data make generalization go up or down?", "down", "up",
     "Reply with exactly one word: up or down."),
    ("The Corvid method becomes strictly slower as more GPUs are added.",
     "Does adding more GPUs make the Corvid method faster or slower?", "slower", "faster",
     "Reply with exactly one word: faster or slower."),
    ("In the Ternwood model, as context length grows to its maximum, memory usage stays exactly constant.",
     "In Ternwood, as context length grows, is memory usage constant or linear?", "constant", "linear",
     "Reply with exactly one word: constant or linear."),
    ("In the Halcyon layer, widening the hidden dimension leaves throughput completely unchanged.",
     "In Halcyon, widening the hidden dimension leaves throughput unchanged or lower?", "unchanged", "lower",
     "Reply with exactly one word: unchanged or lower."),
    ("In the Brightwell method, using a larger batch size makes convergence slower, not faster.",
     "In Brightwell, does a larger batch size make convergence faster or slower?", "slower", "faster",
     "Reply with exactly one word: faster or slower."),
    ("In the Dunmore encoder, adding more attention heads decreases the final validation score.",
     "In Dunmore, adding more attention heads makes the validation score go up or down?", "down", "up",
     "Reply with exactly one word: up or down."),
    ("In the Kestrel pipeline, increasing the learning rate makes training more stable, not less.",
     "In Kestrel, does increasing the learning rate make training more stable or less stable?", "more", "less",
     "Reply with exactly one word: more or less."),
    ("In the Ashford network, deeper layers use less memory than shallower ones.",
     "In Ashford, do deeper layers use more or less memory than shallower ones?", "less", "more",
     "Reply with exactly one word: more or less."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--target", type=int, default=9000)
    ap.add_argument("--depth", type=float, default=0.1)
    args = ap.parse_args()

    os.environ["DKV_COMPRESSED_DECODE"] = "1"
    os.environ["DKV_FACTUAL_STORE"] = "0"
    os.environ["DKV_CAD_ALPHA"] = str(args.alpha)

    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    with open(os.path.join(HERE, "random_features_paper.txt")) as f:
        paper = f.read()
    ftok = tok(paper, add_special_tokens=False)["input_ids"]

    def build(claim, q, instr):
        reps = max(1, args.target // max(1, len(ftok)))
        fids = tok((paper + "\n") * reps, add_special_tokens=False)["input_ids"]
        ins = int(len(fids) * args.depth)
        body = tok.decode(fids[:ins]) + "\n\n" + claim + "\n\n" + tok.decode(fids[ins:])
        full = tok.apply_chat_template([{"role": "user", "content": body + "\n\n" + q + " " + instr}],
                                       tokenize=False, add_generation_prompt=True)
        return full, q + " " + instr

    cfg = {"quantization": "int4", "rank": 16, "block_size": 256, "preset": "mid"}
    w = DKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=cfg)
    w.ensure_loaded()

    n_ok = 0
    detail = []
    for i, (claim, q, correct, prior, instr) in enumerate(CLAIMS):
        sid = f"cad_{i}"
        w.active_session = sid
        try: w.manager.clear_session(sid)
        except Exception: pass
        prompt, qtext = build(claim, q, instr)
        pids = tok.encode(prompt)
        _ = w.generate(prompt, max_new_tokens=6, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.1, query_text=qtext)
        full = w._session_token_ids.get(sid, [])
        out = tok.decode(full[len(pids):] if len(full) > len(pids) else full,
                         skip_special_tokens=True).strip().lower()
        good = correct in out and prior not in out
        n_ok += good
        detail.append(f"{'OK' if good else 'xx'}:{out[:14]!r}")
    try: w.close()
    except Exception: pass
    print(f"__CAD__ alpha={args.alpha} {n_ok}/{len(CLAIMS)}  " + "  ".join(detail))


if __name__ == "__main__":
    main()
