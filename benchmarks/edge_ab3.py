"""Edge-fidelity probe for the TWO reported failure modes (2026-07-14):
  (1) exact LIMITING behavior — "as <param> grows to its maximum, X becomes Y"
      collapses into generic "captures more context / maintains equivariance".
  (2) causal MECHANISM substitution — the exact reason is replaced by a nearby
      invented mechanism (overlapping windows, shifted conv, memory, scalability).

Each claim uses distinctive fictional names + a SPECIFIC mechanism so the model
can't answer from priors, is buried early in real paper prose (must pass through
compressed blocks), and is scored by a right-keyword / wrong-keyword pair where
the wrong set is exactly the generic substitution the user observed.

Arms: dense | sparse_off (REC=0) | sparse_on (REC=1).
"""
import os, sys, json, argparse, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

CLAIMS = [
    dict(kind="limit",
         sent="In the Zarophane network, as the dilation rate grows to the full "
              "input length, each output position's receptive field becomes exactly "
              "the entire input, so the final layer is mathematically equivalent to "
              "one global convolution.",
         q="In the Zarophane network, what does each output position's receptive "
           "field become as the dilation rate grows to the full input length? "
           "Answer in one short sentence.",
         right=r"entire input|whole input|full input|global|equivalent",
         wrong=r"more context|larger|captures more|equivarian|avoids padding|padding"),
    dict(kind="limit",
         sent="In the Brenlow operator, as the number of heads grows to its "
              "maximum, the attention matrix becomes exactly the identity, so the "
              "layer reduces to a plain residual pass-through.",
         q="In the Brenlow operator, what does the attention matrix become as the "
           "number of heads grows to its maximum? Answer in one short sentence.",
         right=r"identity|residual|pass-?through|reduces",
         wrong=r"more context|larger|captures more|richer|more expressive|full rank"),
    dict(kind="cause",
         sent="The Vantrix model needs manual position shifting because its patch "
              "embeddings are tied to absolute coordinates, whereas the Melenor "
              "model needs none because it encodes only relative offsets.",
         q="Why does the Vantrix model need manual position shifting? "
           "Answer in one short sentence.",
         right=r"absolute coordinate|absolute position|tied to absolute",
         wrong=r"overlapping window|shifted conv|memory|scalab|training|inference|efficien"),
    dict(kind="cause",
         sent="The Quorval method avoids retraining because it reuses the frozen "
              "backbone, not because it adds any adapter layers or lowers the "
              "learning rate.",
         q="Why does the Quorval method avoid retraining? Answer in one short sentence.",
         right=r"frozen backbone|reuses.*backbone|frozen",
         wrong=r"adapter|learning rate|smaller|fewer parameter|regulariz"),
]


def load_filler():
    with open(os.path.join(HERE, "random_features_paper.txt")) as f:
        return f.read()


def build_prompt(claim, tokenizer, target_tokens, depth=0.1):
    def templ(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    filler = load_filler()
    question = f"\n\nQuestion: {claim['q']}"
    ov = len(tokenizer(templ(claim["sent"] + question), add_special_tokens=False)["input_ids"])
    ftok = tokenizer(filler, add_special_tokens=False)["input_ids"]
    reps = max(1, (target_tokens - ov) // max(1, len(ftok)))
    fids = tokenizer((filler + "\n") * reps, add_special_tokens=False)["input_ids"]
    ins = int(len(fids) * depth)
    body = tokenizer.decode(fids[:ins]) + "\n\n" + claim["sent"] + "\n\n" + tokenizer.decode(fids[ins:])
    return templ(body + question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dense", "sparse_off", "sparse_on"])
    ap.add_argument("--target", type=int, default=12000)
    ap.add_argument("--gen", type=int, default=44)
    ap.add_argument("--depth", type=float, default=0.1)
    args = ap.parse_args()

    os.environ["DIFFKV_COMPRESSED_DECODE"] = "0" if args.mode == "dense" else "1"
    os.environ["DIFFKV_FACTUAL_STORE"] = "0"
    os.environ["DIFFKV_RESIDUAL_EDGE_CAPTURE"] = "0" if args.mode == "sparse_off" else "1"

    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    cfg = {"quantization": "int4", "rank": 16, "block_size": 256, "preset": "mid"}
    w = DiffKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=cfg)
    w.ensure_loaded()

    results = []
    n_ok = 0
    for i, claim in enumerate(CLAIMS):
        sid = f"e3_{args.mode}_{i}"
        w.active_session = sid
        try: w.manager.clear_session(sid)
        except Exception: pass
        prompt = build_prompt(claim, tok, args.target, args.depth)
        prompt_ids = tok.encode(prompt)
        _ = w.generate(prompt, max_new_tokens=args.gen, temperature=0.0, top_p=1.0,
                       repetition_penalty=1.1, query_text=claim["q"])
        full = w._session_token_ids.get(sid, [])
        gen = full[len(prompt_ids):] if len(full) > len(prompt_ids) else full
        out = tok.decode(gen, skip_special_tokens=True).strip()
        has_r = re.search(claim["right"], out, re.I) is not None
        has_w = re.search(claim["wrong"], out, re.I) is not None
        ok = has_r and not has_w
        n_ok += int(ok)
        results.append(dict(kind=claim["kind"], ok=ok, right=has_r, wrong=has_w, out=out[:150]))

    try: w.close()
    except Exception: pass
    print("__EDGE3__ " + json.dumps(dict(mode=args.mode, ctx=args.target, n_ok=n_ok,
                                         n=len(CLAIMS), results=results)))


if __name__ == "__main__":
    main()
