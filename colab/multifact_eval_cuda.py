#!/usr/bin/env python3
"""Multi-fact accuracy for CUDA DKV vs dense — the coverage this side never had.

benchmarks/rigorous_eval.py holds the only harder-than-NIAH accuracy tests in the
repo (multi-needle, relational binding, synthesis) and imports `mlx.core` in six
places, so NONE of it runs on CUDA. Every accuracy claim on this side therefore
rested on a SINGLE needle, which cannot distinguish the three failure modes that
actually matter:

  multi-needle  retrieval of SEVERAL facts at different depths in one pass. A
                router that keeps one block and drops two scores 1/1 on NIAH and
                1/3 here.
  relational    BINDING: four entities each own a number, and the question names
                one. Retrieving the right region is not enough -- returning
                another entity's number is the characteristic compressed-KV
                failure and NIAH cannot see it at all.
  synthesis     recall spread over a whole document rather than one span, scored
                by how many facts and linked pairs survive.

NEEDLES ARE UNAMBIGUOUS BY CONSTRUCTION. rigorous_eval uses OMEGA-7741-DELTA,
which Qwen splits ' O'|'ME'|'GA', exactly the partial-word shape proven to make
recall a coin flip on small models (validate_cuda_dkv.py's needle note, measured
greedy top-2 margin 0.1875 logits). Reusing those needles here would have
reproduced that contamination in a NEW benchmark. These use one whole-word token
plus single digits, and _assert_needle_unambiguous re-checks it per model at
runtime.

Relational deliberately KEEPS rigorous_eval's invented names (Quillfeather,
Braxanible). They fragment, but the model never has to REGENERATE them -- the
answer is a bare number. What has to be unambiguous is the answer, not the
prompt.

    python colab/multifact_eval_cuda.py --model Qwen/Qwen3.5-2B
    python colab/multifact_eval_cuda.py --model Qwen/Qwen3.5-2B --dense
"""
import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ACTIVE_RUNTIME"))
sys.path.insert(0, _ROOT)

import torch  # noqa: E402

# One whole-word token + digit groups. Verified on Qwen2.5-1.5B and Qwen3.5-2B.
NEEDLES = ["Falcon-9427-6183", "Titan-8305-2194", "Everest-5074-3861"]
NEEDLE_SENTS = [
    "The first secret passcode is Falcon-9427-6183.",
    "The second secret passcode is Titan-8305-2194.",
    "The third secret passcode is Everest-5074-3861.",
]
QUESTION_MULTI = ("What are the three secret passcodes? List them all in order, "
                  "separated by commas.")

FILLER = (
    "The history of artificial intelligence is long and complex. "
    "Early AI researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU compute. "
)

# Verbatim from rigorous_eval.py so the two sides stay comparable.
NATURAL = [("Quillfeather", "4193"), ("Braxanible", "8857"),
           ("Morrowind", "2206"), ("Vantablack", "6034")]
NAT_SENT = ("Dr. {name} reported that the {name}-cluster survey catalogued "
            "precisely {val} variable stars.")
NAT_Q = "How many variable stars did Dr. {name} report?"

FACTS = ["rahimi", "recht", "bochner", "fourier", "sinusoid", "hoeffding",
         "hessian", "laplacian", "binning", "kernel", "support vector machine",
         "least squares", "cvm", "forest", "randomly shifted"]
LINKED_PAIRS = [("rahimi", "recht"), ("bochner", "fourier"),
                ("hoeffding", "convergence"), ("randomly shifted", "binning"),
                ("least squares", "linear")]

_norm = lambda s: "".join(c for c in s.upper() if c.isalnum())  # noqa: E731

_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)


def _assert_needle_unambiguous(tokenizer, needle):
    """Same guard as validate_cuda_dkv.py, per model, before any score is taken."""
    parts = [tokenizer.decode([i]) for i in
             tokenizer(" " + needle, add_special_tokens=False).input_ids]
    words = {w.lower() for w in needle.replace("-", " ").split()}
    bad = [p for p in parts
           if p.strip() and not p.strip().isdigit() and p.strip() not in ("-", "_")
           and p.strip().lower() not in words]
    check(f"needle {needle!r} tokenises unambiguously", not bad,
          f"partial-word pieces {bad} in {parts}" if bad
          else f"{len(parts)} tokens, all whole-word/digit/separator")


class _Dense:
    """Plain HF greedy generation, same surface as the DKV wrapper."""

    def __init__(self, model_id):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="cuda").eval()

    def generate(self, prompt, max_new_tokens, **_kw):
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def _chat(tok, body):
    return tok.apply_chat_template([{"role": "user", "content": body}],
                                   tokenize=False, add_generation_prompt=True)


def _gen(w, prompt, n):
    r = w.generate(prompt=prompt, max_new_tokens=n, temperature=0.0,
                   top_p=1.0, repetition_penalty=1.0)
    return r.rsplit("assistant", 1)[-1].strip()


def _pad(tok, target_tokens):
    """Filler token ids long enough to reach target_tokens."""
    f = tok(FILLER, add_special_tokens=False).input_ids
    return (f * (target_tokens // max(1, len(f)) + 2))[:target_tokens]


def test_multi_needle(w, ctx):
    tok = w.tokenizer
    body_ids = _pad(tok, ctx)
    # Three needles at 25/50/75% depth -- a router that keeps only the most
    # recent, or only the first, is visible here and invisible on single-needle.
    outp = []
    marks = [int(len(body_ids) * d) for d in (0.25, 0.50, 0.75)]
    prev = 0
    for m, sent in zip(marks, NEEDLE_SENTS):
        outp.append(tok.decode(body_ids[prev:m]))
        outp.append(" " + sent + " ")
        prev = m
    outp.append(tok.decode(body_ids[prev:]))
    body = "".join(outp) + "\n\n" + QUESTION_MULTI
    prompt = _chat(tok, body)
    ntok = len(tok(prompt).input_ids)

    out = _gen(w, prompt, 64)
    got = _norm(out)
    found = [n for n in NEEDLES if _norm(n) in got]
    check(f"multi-needle {ctx//1024}k — all 3 passcodes", len(found) == 3,
          f"{len(found)}/3 found; missing="
          f"{[n for n in NEEDLES if n not in found]} out={out[:80]!r}")
    return len(found), 3


def test_relational(w, ctx):
    """Entity->value BINDING. Returning another entity's number is the failure
    mode single-needle cannot express, so score every entity, not just one."""
    tok = w.tokenizer
    body_ids = _pad(tok, ctx)
    marks = [int(len(body_ids) * d) for d in (0.15, 0.38, 0.61, 0.84)]
    parts, prev = [], 0
    for m, (name, val) in zip(marks, NATURAL):
        parts.append(tok.decode(body_ids[prev:m]))
        parts.append(" " + NAT_SENT.format(name=name, val=val) + " ")
        prev = m
    parts.append(tok.decode(body_ids[prev:]))
    base = "".join(parts)

    hits = 0
    for name, val in NATURAL:
        body = base + "\n\nQuestion: " + NAT_Q.format(name=name) + " Answer with the number only."
        out = _gen(w, _chat(tok, body), 24)
        digits = re.findall(r"\d{3,}", out)
        ok = bool(digits) and digits[0] == val
        # Report WHICH value came back: another entity's number is a binding
        # failure, a number belonging to nobody is a retrieval failure, and the
        # two want different fixes.
        other = next((n for n, v in NATURAL if digits and digits[0] == v and n != name), None)
        check(f"relational {ctx//1024}k — Dr. {name} -> {val}", ok,
              f"got {digits[:1] or out[:40]!r}"
              + (f"  (that is Dr. {other}'s value — BINDING failure)" if other else ""))
        hits += int(ok)
    return hits, len(NATURAL)


def test_synthesis(w, ctx):
    """Whole-document recall, scored by surviving facts and linked pairs."""
    # MUST be the Random Features paper: the FACTS/LINKED_PAIRS lists above are
    # rigorous_eval's and are drawn from it (rahimi, recht, bochner, fourier...).
    # Scoring them against a different document measures nothing -- pointed at
    # nat_paper.txt this returned 1/15 facts on the DENSE control, which reads as
    # a catastrophic engine failure and is really just the wrong corpus.
    # Verified: random_features_paper.txt contains 6/6 of the probe words,
    # nat_paper.txt contains 0/6.
    paper = os.path.join(_ROOT, "benchmarks", "random_features_paper.txt")
    if not os.path.exists(paper):
        check(f"synthesis {ctx//1024}k", False, f"corpus missing: {paper}")
        return 0, 1
    tok = w.tokenizer
    text = open(paper, encoding="utf-8", errors="ignore").read()
    ids = tok(text, add_special_tokens=False).input_ids[:ctx]
    body = tok.decode(ids) + (
        "\n\nSummarise the key technical contributions of this paper, naming the "
        "authors, the mathematical results it relies on, and the methods it is "
        "compared against.")
    out = _gen(w, _chat(tok, body), 320).lower()
    n_facts = sum(1 for f in FACTS if f in out)
    n_link = sum(1 for a, b in LINKED_PAIRS if a in out and b in out)
    score = (n_facts / len(FACTS)) * 50.0 + (n_link / len(LINKED_PAIRS)) * 50.0
    # Threshold is a REGRESSION GATE, not a quality bar: it exists so a change
    # that halves document-level recall fails loudly. Report the raw score too.
    check(f"synthesis {ctx//1024}k — score >= 30", score >= 30.0,
          f"score={score:.1f} (facts {n_facts}/{len(FACTS)}, "
          f"links {n_link}/{len(LINKED_PAIRS)})")
    return n_facts, len(FACTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--dense", action="store_true", help="plain HF, DKV not loaded")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--tests", default="multi_needle,relational,synthesis")
    a = ap.parse_args()

    os.environ.setdefault("DKV_ENGAGE_THRESHOLD", "1024")
    if a.dense:
        w = _Dense(a.model)
        print(f"=== DENSE control — {a.model} ===", flush=True)
    else:
        from serving.decode_config import BEST_DECODE_DEFAULTS
        for k, v in BEST_DECODE_DEFAULTS.items():
            os.environ.setdefault(k, v)
        from ACTIVE_RUNTIME.serving.hf_dkv_wrapper import PyTorchDKVHFWrapper
        w = PyTorchDKVHFWrapper(model_id=a.model, config={"mode": "fp16", **({"micro_block_size": int(os.environ["BLOCK"])} if os.environ.get("BLOCK") else {}),
                                **({"rank": int(os.environ["RANK"])} if os.environ.get("RANK") else {})},
                                device="cuda")
        w.ensure_loaded()
        print(f"=== DKV — {a.model} ===", flush=True)

    for n in NEEDLES:
        _assert_needle_unambiguous(w.tokenizer, n)

    want = set(a.tests.split(","))
    if "multi_needle" in want:
        test_multi_needle(w, a.ctx)
    if "relational" in want:
        test_relational(w, a.ctx)
    if "synthesis" in want:
        test_synthesis(w, a.ctx)

    n_ok = sum(1 for _, ok in _RESULTS if ok)
    print(f"\n=== {n_ok}/{len(_RESULTS)} checks passed "
          f"({'DENSE' if a.dense else 'DKV'}, ctx={a.ctx}) ===", flush=True)
    sys.exit(0 if n_ok == len(_RESULTS) else 1)


if __name__ == "__main__":
    main()
