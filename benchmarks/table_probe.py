#!/usr/bin/env python3
"""Table-row binding probe (ACTIVE_RUNTIME / MLX).

The binding_probe plants prose facts ("The X facility processes N samples per
day"); this probe plants a TABLE — the failure surface reported 2026-07-12:
DKV retains concepts and values from a paper's ablation tables (NA,
Swin-Tiny, kernel sizes, 83.2, throughput) but re-attaches them to the wrong
rows/metrics (83.2 moved from 7x7 to 3x3, a fabricated 4x4 row, imgs/sec
replaced by invented G/s numbers).

Tables are structurally hostile to the current residual capture: nearly every
token in a row is is_core (digits) or non-prose (pipes), so whole rows form one
giant boosted segment competing for residual slots, and the row key (7x7) is
numeric — the owner-capture walk (which looks for a capitalized word) never
fires for it. Column headers live far away, often in another block.

We plant a 4-row x 2-metric kernel-size ablation table mid-depth in filler and
ask: reproduce-table (list), row->value (fwd), value->row (rev). Score per row:
correct / swap (another planted value) / miss, plus fabricated-value and
fabricated-row detection.

Usage:
  python3 benchmarks/table_probe.py --ctx 8192 --modes dense dkv
  python3 benchmarks/table_probe.py --ctx 8192 --modes dkv --full-matrix
(Each cell runs in its own subprocess; the driver aggregates.)
"""
import os, sys, json, argparse, subprocess, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")

# (row_key, top1, throughput) — distinct values, no substring collisions.
ROWS = [
    ("3x3", "81.4", "1350"),
    ("5x5", "82.6", "1201"),
    ("7x7", "83.2", "1136"),
    ("9x9", "83.0", "1042"),
]
TOP1_VALS = {r[1] for r in ROWS}
THR_VALS = {r[2] for r in ROWS}
ALL_VALS = TOP1_VALS | THR_VALS
# Row keys the model might fabricate (seen in the wild: 4x4).
FABRICATED_KEYS = ["1x1", "2x2", "4x4", "6x6", "8x8", "11x11", "13x13"]

TABLE_TEXT = (
    "Table 4 reports the kernel size ablation for the NA-Tiny model on "
    "ImageNet-1K, comparing against the Swin-Tiny baseline with overlapping "
    "convolutions in the tokenizer.\n\n"
    "| Kernel size | Top-1 accuracy (%) | Throughput (imgs/sec) |\n"
    "|-------------|--------------------|------------------------|\n"
    + "".join(f"| {k} | {t} | {p} |\n" for k, t, p in ROWS)
    + "\nThe 7x7 kernel offers the best accuracy-throughput trade-off and is "
    "used in all NA-Tiny experiments.\n"
)

FILLER_PARA = (
    "The history of artificial intelligence is long and complex. Early AI "
    "researchers believed that machines could simulate human reasoning. "
    "Symbolic AI dominated the field for decades, followed by neural networks. "
    "Deep learning transformed AI in the 2010s with massive datasets and GPU "
    "compute. "
)

SYSTEM_PART = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
ASSIST_PART = "<|im_end|>\n<|im_start|>assistant\n"

LIST_Q = ("Based on Table 4 above, reproduce the kernel size ablation exactly: "
          "for each kernel size, state its Top-1 accuracy and throughput, one "
          "per line, in the format 'KxK: top1=NUMBER, throughput=NUMBER'.")

def fwd_q(key):
    return (f"According to Table 4 above, what is the Top-1 accuracy of the "
            f"{key} kernel size? Answer with only the number.")

def thr_q(key):
    return (f"According to Table 4 above, what is the throughput in imgs/sec "
            f"of the {key} kernel size? Answer with only the number.")

def rev_q(val):
    return (f"According to Table 4 above, which kernel size achieves {val}% "
            f"Top-1 accuracy? Answer with only the kernel size.")

def _key_pat(key):
    a, b = key.split("x")
    return re.compile(rf"\b{a}\s*[x×]\s*{b}\b", re.IGNORECASE)

_NUM = re.compile(r"\d+(?:\.\d+)?")

def score_list(text):
    """Per row: correct (line with its key contains its own top1),
    swap (contains another planted value instead), fab (contains a number not
    in the planted set), miss. Plus fabricated row keys."""
    res = {}
    lines = text.splitlines()
    for key, top1, thr in ROWS:
        verdict = "miss"
        for ln in lines:
            if _key_pat(key).search(ln):
                nums = set(_NUM.findall(ln.replace(",", "")))
                nums -= {key.split("x")[0], key.split("x")[1]}
                if top1 in nums and (thr in nums or not (nums & THR_VALS - {thr})):
                    verdict = "correct" if thr in nums else "top1_only"
                elif nums & ALL_VALS:
                    verdict = "swap"
                elif nums:
                    verdict = "fab"
                break
        res[key] = verdict
    fab_rows = [k for k in FABRICATED_KEYS
                if any(_key_pat(k).search(ln) for ln in lines)]
    return res, fab_rows

def classify_num(answer, expect, others):
    nums = set(_NUM.findall(answer.replace(",", "")))
    if expect in nums:
        return "correct"
    if nums & set(others):
        return "swap"
    return "fab" if nums else "miss"

def classify_key(answer, expect):
    if _key_pat(expect).search(answer):
        return "correct"
    for key, _, _ in ROWS:
        if key != expect and _key_pat(key).search(answer):
            return "swap"
    for key in FABRICATED_KEYS:
        if _key_pat(key).search(answer):
            return "fab"
    return "miss"

def build_prompt(tokenizer, ctx, question):
    sys_ids = tokenizer.encode(SYSTEM_PART, add_special_tokens=False)
    q_ids = tokenizer.encode("\n\n" + question + ASSIST_PART, add_special_tokens=False)
    table_ids = tokenizer.encode("\n\n" + TABLE_TEXT + "\n", add_special_tokens=False)
    budget = ctx - len(sys_ids) - len(q_ids) - len(table_ids)
    filler_ids = tokenizer.encode(FILLER_PARA, add_special_tokens=False)
    n_rep = max(1, budget // len(filler_ids) + 1)
    filler_all = (filler_ids * n_rep)[:budget]
    cut = budget // 2  # table sits at depth 0.5
    ids = sys_ids + filler_all[:cut] + table_ids + filler_all[cut:] + q_ids
    return tokenizer.decode(ids)

def run_cell(mode, ctx, question, max_tokens):
    """Executed inside the isolated subprocess (same harness as binding_probe)."""
    import torch
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    os.environ["DKV_COMPRESSED_DECODE"] = "0" if mode == "dense" else "1"
    os.environ["DKV_LEGO_PREFILL"] = "1" if mode == "lego" else "0"
    from serving.hf_dkv_wrapper import DKVHFWrapper
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DKVHFWrapper(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", config=cfg)
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model
    prompt = build_prompt(tok, ctx, question)
    ids = tok.encode(prompt)
    sid = "table_probe"
    mgr.clear_session(sid)
    wrapper._session_token_ids[sid] = []
    mgr.init_session(sid, prefill_len=len(ids))
    mgr.register_prefill_tokens(sid, torch.tensor(ids, dtype=torch.long))
    model._dkv_session_ids = [sid]
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
    ap.add_argument("--modes", nargs="+", default=["dense", "dkv"])
    ap.add_argument("--full-matrix", action="store_true",
                    help="also run per-row fwd/thr/rev questions (slow)")
    ap.add_argument("--single", nargs=3, metavar=("MODE", "KIND", "ARG"))
    args = ap.parse_args()

    if args.single:
        mode, kind, arg = args.single
        if kind == "list":
            run_cell(mode, args.ctx, LIST_Q, 180)
        elif kind == "fwd":
            run_cell(mode, args.ctx, fwd_q(arg), 16)
        elif kind == "thr":
            run_cell(mode, args.ctx, thr_q(arg), 16)
        else:
            val = {k: t for k, t, _ in ROWS}[arg]
            run_cell(mode, args.ctx, rev_q(val), 16)
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

    summary = {}
    for mode in args.modes:
        t0 = time.time()
        print(f"\n=== MODE {mode} ctx={args.ctx} ===", flush=True)
        ans = spawn(mode, "list", "-")
        table, fab_rows = score_list(ans)
        counts = {k: sum(1 for v in table.values() if v == k)
                  for k in ("correct", "top1_only", "swap", "fab", "miss")}
        print(f"LIST [{mode}]: {counts}  fabricated_rows={fab_rows}  {table}")
        print(f"  output: {ans[:400]!r}")
        summary[mode] = {"list": counts, "fab_rows": fab_rows}
        if args.full_matrix:
            for kind, expect_fn, others_fn, cls in (
                ("fwd", lambda r: r[1], lambda r: TOP1_VALS - {r[1]}, "num"),
                ("thr", lambda r: r[2], lambda r: THR_VALS - {r[2]}, "num"),
                ("rev", lambda r: r[0], None, "key"),
            ):
                res = []
                for row in ROWS:
                    a = spawn(mode, kind, row[0])
                    if cls == "num":
                        v = classify_num(a, expect_fn(row), others_fn(row))
                    else:
                        v = classify_key(a, row[0])
                    res.append((row[0], v, a.strip()[:40]))
                cnt = {k: sum(1 for _, v, _ in res if v == k)
                       for k in ("correct", "swap", "fab", "miss")}
                print(f"{kind.upper()} [{mode}]: {cnt}")
                for r_ in res: print(f"   {r_}")
                summary[mode][kind] = cnt
        print(f"  mode wall time {time.time()-t0:.0f}s", flush=True)

    print("\n===== TABLE PROBE SUMMARY =====")
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
