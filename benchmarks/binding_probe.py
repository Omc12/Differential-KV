#!/usr/bin/env python3
"""Relational-binding probe (ACTIVE_RUNTIME / MLX).

Distinguishes RETENTION failure (fact missing) from BINDING failure (the model
retrieves planted nouns and numbers but attaches the wrong number to the noun).
Six entity->value sentences are planted at spread depths in filler; we then ask
- list-all: "list each facility and its daily sample count" (one generation,
  score the reconstructed table: correct / SWAPPED / missing per entity), and
- per-question: forward ("how many does X process?") and reverse ("which
  facility processes N?") for a subset of entities, one isolated run each.

An answer that names ANOTHER planted value/entity is a SWAP (binding failure);
an answer with none of the planted set is a MISS (retention/other). If swaps
dominate misses, the fix is relational locality, not wider capture.

Modes: dense (COMPRESSED_DECODE=0), diffkv (compressed, DIFFKV_LEGO_PREFILL=0),
lego (compressed, DIFFKV_LEGO_PREFILL=1 — studs default on MLX).

Usage:
  python3 benchmarks/binding_probe.py --ctx 8192 --modes dense diffkv lego
  python3 benchmarks/binding_probe.py --ctx 8192 --modes lego --full-matrix
(Each cell runs in its own subprocess; the driver aggregates.)
"""
import os, sys, json, argparse, subprocess, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

ENTITIES = [
    ("Meridian",  "4382"),
    ("Okazaki",   "7156"),
    ("Halvorsen", "2903"),
    ("Brancusi",  "8617"),
    ("Tarkovsky", "5248"),
    ("Ellsworth", "1794"),
]
# Which entities the per-question matrix probes (middle depths — hardest).
MATRIX_ENTITIES = ["Okazaki", "Halvorsen", "Brancusi", "Tarkovsky"]

FILLER_PARA = (
    "The history of artificial intelligence is long and complex. Early AI "
    "researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU "
    "compute. "
)

SYSTEM_PART = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
ASSIST_PART = "<|im_end|>\n<|im_start|>assistant\n"

LIST_Q = ("Based on the facility reports above, list each facility and its "
          "daily sample count, one per line, in the format 'Name: number'.")

def fwd_q(name):
    return (f"According to the facility reports above, how many samples per day "
            f"does the {name} facility process? Answer with only the number.")

def rev_q(value):
    return (f"According to the facility reports above, which facility processes "
            f"{value} samples per day? Answer with only the facility name.")

# Comparison question — RC5/RC8's actual design target: two entities whose
# property values can interleave/invert. A binding failure here emits e.g.
# "Okazaki processes 2903" (Halvorsen's value) — the relationship inversion.
def cmp_q(a, b):
    return (f"According to the facility reports above, compare the {a} and {b} "
            f"facilities. State each facility's exact daily sample count, one "
            f"per line as 'Name: number'.")

def score_cmp(text, a, b):
    """Per named entity: correct / swap / miss (same logic as list-all but only
    for the two compared entities)."""
    vals = dict(ENTITIES)
    lines = text.lower().splitlines()
    allvals = {v for _, v in ENTITIES}
    out = {}
    for name in (a, b):
        val = vals[name]
        verdict = "miss"
        for ln in lines:
            if re.search(_name_pat(name), ln):
                nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", ln)}
                planted = nums & allvals
                if val in planted:
                    verdict = "correct"
                elif planted:
                    verdict = "swap"
                break
        out[name] = verdict
    return out

def build_prompt(tokenizer, ctx, question):
    sents = [f"The {n} facility processes {v} samples per day." for n, v in ENTITIES]
    sys_ids = tokenizer.encode(SYSTEM_PART, add_special_tokens=False)
    q_ids = tokenizer.encode("\n\n" + question + ASSIST_PART, add_special_tokens=False)
    sent_ids = [tokenizer.encode(" " + s, add_special_tokens=False) for s in sents]
    budget = ctx - len(sys_ids) - len(q_ids) - sum(len(x) for x in sent_ids)
    filler_ids = tokenizer.encode(FILLER_PARA, add_special_tokens=False)
    n_rep = max(1, budget // len(filler_ids) + 1)
    filler_all = (filler_ids * n_rep)[:budget]
    # interleave: needle i sits at depth (i+1)/(n+1) of the filler budget
    n = len(sents)
    segs, prev = [], 0
    body_ids = []
    for i in range(n):
        cut = int(budget * (i + 1) / (n + 1))
        body_ids += filler_all[prev:cut] + sent_ids[i]
        prev = cut
    body_ids += filler_all[prev:]
    ids = sys_ids + body_ids + q_ids
    return tokenizer.decode(ids)

def _name_pat(name):
    # tolerate plural/possessive surface artifacts ("Meridians", "Okazakis")
    return r"\b" + re.escape(name.lower()) + r"s?\b"

def classify(answer, expect, others):
    """correct / swap (another planted item) / miss."""
    a = answer.lower()
    if re.search(_name_pat(expect), a):
        return "correct"
    for o in others:
        if re.search(_name_pat(o), a):
            return "swap"
    return "miss"

def score_list_all(text):
    """Per entity: correct (name near its value), swap (name near another
    planted value), miss (name absent or no planted value near it)."""
    res = {}
    lines = text.lower().splitlines()
    values = {v for _, v in ENTITIES}
    for name, val in ENTITIES:
        verdict = "miss"
        for ln in lines:
            if re.search(_name_pat(name), ln):
                nums = set(re.findall(r"\d[\d,]*", ln))
                nums = {n.replace(",", "") for n in nums}
                planted = nums & values
                if val in planted:
                    verdict = "correct"
                elif planted:
                    verdict = "swap"
                break
        res[name] = verdict
    return res

def run_cell(mode, ctx, question, max_tokens):
    """Executed inside the isolated subprocess."""
    import torch
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    os.environ["DIFFKV_COMPRESSED_DECODE"] = "0" if mode == "dense" else "1"
    os.environ["DIFFKV_LEGO_PREFILL"] = "1" if mode == "lego" else "0"
    from serving.hf_diffkv_wrapper import DiffKVHFWrapper
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DiffKVHFWrapper(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", config=cfg)
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model
    prompt = build_prompt(tok, ctx, question)
    ids = tok.encode(prompt)
    sid = "binding_probe"
    mgr.clear_session(sid)
    wrapper._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    model._diffkv_session_ids = [sid]
    import numpy as np
    CH = 512
    output = None
    for cs in range(0, len(ids), CH):
        chunk = ids[cs:cs + CH]
        ct = torch.tensor([chunk], dtype=torch.long)
        pt = torch.tensor([list(range(cs, cs + len(chunk)))], dtype=torch.long)
        output = model(ct, pt)
        mgr.compress_deferred_prefill_blocks(sid)
    logits = output.logits[0, -1].cpu().numpy()
    cur = len(ids)
    generated = []
    for _ in range(max_tokens):
        nid = int(np.argmax(logits))
        generated.append(nid)
        if nid == tok.eos_token_id:
            break
        mgr.register_prefill_tokens(sid, torch.tensor([nid], dtype=torch.long))
        output = model(torch.tensor([[nid]], dtype=torch.long),
                       torch.tensor([[cur]], dtype=torch.long))
        logits = output.logits[0, -1].cpu().numpy()
        cur += 1
    print(json.dumps({"answer": tok.decode(generated)}))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--modes", nargs="+", default=["dense", "diffkv", "lego"])
    ap.add_argument("--full-matrix", action="store_true",
                    help="also run the per-question fwd/rev matrix (slow)")
    ap.add_argument("--single", nargs=3, metavar=("MODE", "KIND", "ARG"),
                    help="internal: run one cell (KIND=list|fwd|rev, ARG=entity or '-')")
    args = ap.parse_args()

    if args.single:
        mode, kind, arg = args.single
        if kind == "list":
            run_cell(mode, args.ctx, LIST_Q, 140)
        elif kind == "fwd":
            run_cell(mode, args.ctx, fwd_q(arg), 24)
        else:
            val = dict(ENTITIES)[arg]
            run_cell(mode, args.ctx, rev_q(val), 24)
        return

    def spawn(mode, kind, arg):
        cmd = [sys.executable, os.path.abspath(__file__), "--ctx", str(args.ctx),
               "--single", mode, kind, arg]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("{"):
                try:
                    return json.loads(ln)["answer"]
                except Exception:
                    pass
        print(f"  [cell {mode}/{kind}/{arg} failed]\n{(r.stderr or '')[-500:]}", file=sys.stderr)
        return ""

    names = [n for n, _ in ENTITIES]
    values = [v for _, v in ENTITIES]
    summary = {}
    for mode in args.modes:
        t0 = time.time()
        print(f"\n=== MODE {mode} ctx={args.ctx} ===", flush=True)
        ans = spawn(mode, "list", "-")
        table = score_list_all(ans)
        c = sum(1 for v in table.values() if v == "correct")
        s = sum(1 for v in table.values() if v == "swap")
        m = sum(1 for v in table.values() if v == "miss")
        print(f"LIST-ALL [{mode}]: correct={c} swap={s} miss={m}  {table}")
        print(f"  output: {ans[:300]!r}")
        summary[mode] = {"list": (c, s, m)}
        if args.full_matrix:
            fwd_res, rev_res = [], []
            for name in MATRIX_ENTITIES:
                val = dict(ENTITIES)[name]
                a1 = spawn(mode, "fwd", name)
                v1 = classify(a1, val, [v for v in values if v != val])
                fwd_res.append((name, v1, a1.strip()[:60]))
                a2 = spawn(mode, "rev", name)
                v2 = classify(a2, name, [n for n in names if n != name])
                rev_res.append((name, v2, a2.strip()[:60]))
            fc = {k: sum(1 for _, v, _ in fwd_res if v == k) for k in ("correct", "swap", "miss")}
            rc = {k: sum(1 for _, v, _ in rev_res if v == k) for k in ("correct", "swap", "miss")}
            print(f"FWD (entity->value) [{mode}]: {fc}")
            for r_ in fwd_res: print(f"   {r_}")
            print(f"REV (value->entity) [{mode}]: {rc}")
            for r_ in rev_res: print(f"   {r_}")
            summary[mode]["fwd"] = fc
            summary[mode]["rev"] = rc
        print(f"  mode wall time {time.time()-t0:.0f}s", flush=True)

    print("\n===== BINDING PROBE SUMMARY =====")
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
