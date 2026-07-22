"""
Relational / multi-entity A/B eval for the MLX ACTIVE runtime.

Unlike the single-needle NIAH (which the residual exact-token capture already
nails), this stresses MULTI-ENTITY BINDING: several modules each have a distinct
activation key that all share a "BRAVO-" prefix, so the keys interfere. To answer
"what is module X's key" the model must bind the RIGHT number to the RIGHT module
name — exactly the case the factual store's entity machinery is meant to fix and
that SVD+residuals alone can blur.

Three arms (selected by --mode, which sets the env the manager reads):
  exact          : DKV_COMPRESSED_DECODE=0  DKV_FACTUAL_STORE=0  (full-KV upper bound)
  sparse         : DKV_COMPRESSED_DECODE=1  DKV_FACTUAL_STORE=0  (does compression mis-bind?)
  sparse_factual : DKV_COMPRESSED_DECODE=1  DKV_FACTUAL_STORE=1  (does the store restore it?)

Facts are placed EARLY and padded with filler so they fall outside the dense
recency window (i.e. they must be reached through compressed blocks, where the
factual store can matter).
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

# Modules + interfering keys (shared BRAVO- prefix; distinguishing 4-digit tail).
MODULES = [
    ("Wren",   "BRAVO-2741"),
    ("Heron",  "BRAVO-5198"),
    ("Falcon", "BRAVO-8853"),
    ("Osprey", "BRAVO-3306"),
    ("Raven",  "BRAVO-6620"),
]

# Natural-prose variant (RC4-style): rare distinctive NAMES + a distinct value, each
# in a flowing sentence so the entity name and its value sit in the same salient span.
# This is the layout relational binding was designed for (vs the adversarial registry).
NATURAL = [
    ("Quillfeather", "4193"),
    ("Braxanible",   "8857"),
    ("Morrowind",    "2206"),
    ("Vantablack",   "6034"),
]
NAT_SENT = "Dr. {name} reported that the {name}-cluster survey catalogued precisely {val} variable stars."
NAT_Q = "How many variable stars did Dr. {name} report?"

FILLER = (
    "The facility maintenance log records routine calibration of the cooling "
    "loops, periodic inspection of the conduit seals, and scheduled rotation of "
    "the backup generators. Technicians note ambient humidity, verify the airlock "
    "interlocks, and confirm that the telemetry uplink remains within nominal "
    "bounds throughout each shift. None of these housekeeping entries alter the "
    "module registry described above. "
)


def build_prompt(ask_module, tokenizer, target_tokens=3500, spread=False):
    question = (
        f"\n\nQuestion: State ONLY the exact activation key of the {ask_module} module, "
        f"copied verbatim from the registry above. Answer with the key and nothing else."
    )

    def n(t):
        return len(tokenizer(t, add_special_tokens=False)["input_ids"])

    def templ(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    filler_tok = max(1, n(FILLER))
    if spread:
        # One fact per region, each separated by ~1 block of filler so the facts
        # land in DIFFERENT compressed blocks (each gets the full residual budget).
        fact_lines = [f"For the record, the {name} module's activation key is {key}.\n"
                      for name, key in MODULES]
        per_gap = max(1, (target_tokens // (len(MODULES) + 1)) // filler_tok)
        body = "Facility log.\n"
        for fl in fact_lines:
            body += (FILLER * per_gap) + fl
        body += FILLER * per_gap
        return templ(body + question)
    facts = "Module activation registry (memorize each module's exact key):\n" + "".join(
        f"- The {name} module's activation key is {key}.\n" for name, key in MODULES
    ) + "\n"
    overhead = n(templ(facts + question))
    reps = max(1, (target_tokens - overhead) // filler_tok)
    body = facts + (FILLER * reps) + question
    return templ(body)


def build_prompt_natural(ask_name, tokenizer, target_tokens=6000, spread=True):
    """RC4-style: rare distinctive names + values in flowing sentences, so each
    entity name sits in the same salient span as its value."""
    question = "\n\nQuestion: " + NAT_Q.format(name=ask_name) + " Answer with the number only."

    def n(t):
        return len(tokenizer(t, add_special_tokens=False)["input_ids"])

    def templ(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    fact_lines = [NAT_SENT.format(name=nm, val=v) + " " for nm, v in NATURAL]
    filler_tok = max(1, n(FILLER))
    if spread:
        per_gap = max(1, (target_tokens // (len(NATURAL) + 1)) // filler_tok)
        body = "Observatory bulletin.\n"
        for fl in fact_lines:
            body += (FILLER * per_gap) + fl + "\n"
        body += FILLER * per_gap
    else:
        facts = "Observatory bulletin.\n" + "".join(fl + "\n" for fl in fact_lines) + "\n"
        overhead = n(templ(facts + question))
        reps = max(1, (target_tokens - overhead) // filler_tok)
        body = facts + (FILLER * reps)
    return templ(body + question)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["exact", "sparse", "sparse_factual"])
    ap.add_argument("--target", type=int, default=3500)
    ap.add_argument("--gen", type=int, default=24)
    ap.add_argument("--spread", action="store_true",
                    help="place one fact per block (each gets full residual budget)")
    ap.add_argument("--natural", action="store_true",
                    help="RC4-style natural-prose entities (rare names + value in one sentence)")
    args = ap.parse_args()

    # Env MUST be set before the wrapper/manager is constructed.
    os.environ["DKV_COMPRESSED_DECODE"] = "0" if args.mode == "exact" else "1"
    os.environ["DKV_FACTUAL_STORE"] = "1" if args.mode == "sparse_factual" else "0"

    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    cfg = {"quantization": "int4", "rank": int(os.environ.get("DKV_RANK", "16")),
           "block_size": 256, "preset": "mid"}
    w = DKVHFWrapper(model_id="Qwen/Qwen2.5-1.5B-Instruct", config=cfg)
    w.ensure_loaded()

    entities = NATURAL if args.natural else MODULES

    results = []
    n_correct = n_misbound = 0
    for i, (name, key) in enumerate(entities):
        sid = f"rel_{args.mode}_{i}"
        w.active_session = sid
        try:
            w.manager.clear_session(sid)
        except Exception:
            pass
        if args.natural:
            prompt = build_prompt_natural(name, tok, args.target, spread=args.spread)
            qtext = NAT_Q.format(name=name)
        else:
            prompt = build_prompt(name, tok, args.target, spread=args.spread)
            qtext = f"State the exact activation key of the {name} module."
        prompt_ids = tok.encode(prompt)
        _ = w.generate(prompt, max_new_tokens=args.gen, temperature=0.1,
                       top_p=1.0, repetition_penalty=1.1, query_text=qtext)
        # generate() returns prompt+generation decoded; the prompt echoes EVERY key,
        # so score ONLY the newly generated tokens (stored on the session).
        full_ids = w._session_token_ids.get(sid, [])
        gen_ids = full_ids[len(prompt_ids):] if len(full_ids) > len(prompt_ids) else full_ids
        out = tok.decode(gen_ids, skip_special_tokens=True)
        correct = key in out
        # Also score the bare VALUE (distinguishing digits) — credits correct
        # entity→value binding even when the shared prefix is mangled or the value
        # repeats (a generation-quality artifact, not a binding error).
        num = key.split("-")[-1]
        num_correct = num in out
        others = [k for nm, k in entities if k != key]
        other_nums = [k.split("-")[-1] for k in others]
        misbound = (not num_correct) and any(o in out for o in other_nums)
        n_correct += int(correct)
        n_misbound += int(misbound)
        results.append({"module": name, "want": key, "correct": correct,
                        "num_correct": num_correct, "misbound": misbound,
                        "out": out.strip()[:80]})

    try:
        w.close()
    except Exception:
        pass

    n_num_correct = sum(int(r["num_correct"]) for r in results)
    summary = {"mode": args.mode, "n_total": len(entities),
               "n_correct": n_correct, "n_num_correct": n_num_correct,
               "n_misbound": n_misbound, "results": results}
    print("__RELAB__ " + json.dumps(summary))


if __name__ == "__main__":
    main()
