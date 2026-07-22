#!/usr/bin/env python3
"""Table-row binding probe v2 — realistic conditions (ACTIVE_RUNTIME / MLX).

v1 (table_probe.py) planted a small 4-row table in REPEATED fluffy filler at 8k
and found dkv == dense everywhere. But the reported failure (2026-07-12,
NAT-style paper: 83.2 moved from 7x7 to 3x3, fabricated 4x4 row, imgs/sec ->
invented G/s) happened on a real paper. v2 models what v1 didn't:

  1. REAL technical filler (random_features_paper.txt cycled) — every block
     has digits/math competing for residual slots, and the word 'kernel'
     saturates lexical routing signals;
  2. a BIG table (6 rows x 3 metrics, ~300 tokens) deliberately STRADDLING a
     256-token block boundary — a split row has its key in block A and its
     value in block B;
  3. a SECOND table (ablation path Swin-T -> ... -> NAT-T) so values can
     migrate across tables;
  4. ctx 8192/16384 — routing retrieves topk=16 of 32/64 blocks.

Usage:
  python3 benchmarks/table_probe2.py --ctx 16384 --modes dense dkv
  python3 benchmarks/table_probe2.py --ctx 16384 --modes dkv \
      --extra-env DKV_MAX_RESIDUAL=256 DKV_TOPK_BLOCKS=32
(Each cell runs in its own subprocess; the driver aggregates.)
"""
import os, sys, json, argparse, subprocess, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
FILLER_FILE = os.path.join(HERE, "random_features_paper.txt")
BLOCK = 256

# ── TABLE A: kernel-size ablation, 6 rows x (Top-1, Top-5, throughput) ──
ROWS_A = [
    # key,   top1,   top5,   imgs/sec
    ("3x3",   "79.1", "94.2", "1512"),
    ("5x5",   "80.3", "94.8", "1377"),
    ("7x7",   "81.7", "95.3", "1245"),
    ("9x9",   "82.9", "95.9", "1093"),
    ("11x11", "82.4", "95.6",  "987"),
    ("13x13", "81.9", "95.1",  "902"),
]
A_TOP1 = {r[1] for r in ROWS_A}
A_TOP5 = {r[2] for r in ROWS_A}
A_THR = {r[3] for r in ROWS_A}
FABRICATED_KEYS = ["1x1", "2x2", "4x4", "6x6", "8x8", "10x10", "12x12", "15x15"]

_A_CAPTION = (
    "Table 4 reports the kernel size ablation for the NAT-Tiny model on "
    "ImageNet-1K. All variants use the same overlapping convolutional "
    "tokenizer and training recipe; only the neighborhood attention kernel "
    "size changes.\n\n"
)
_A_CLOSE = ("\nThe 9x9 kernel achieves the best Top-1 accuracy, while 3x3 "
            "gives the highest throughput.\n")

# --style markdown (default): pipe-delimited, what a .md source looks like.
# --style aligned: what a PDF copy-paste looks like — whitespace-separated
#   columns, one line per row, x rendered as the multiplication glyph, no
#   pipes anywhere. --style flat: the worst PDF extraction — the whole table
#   body runs together on a single line.
def _table_a_text(style):
    if style == "markdown":
        return (_A_CAPTION
                + "| Kernel size | Top-1 (%) | Top-5 (%) | Throughput (imgs/sec) |\n"
                + "|-------------|-----------|-----------|------------------------|\n"
                + "".join(f"| {k} | {t1} | {t5} | {p} |\n" for k, t1, t5, p in ROWS_A)
                + _A_CLOSE)
    gl = lambda k: k.replace("x", "×")
    if style == "aligned":
        return (_A_CAPTION
                + "Kernel size   Top-1 (%)   Top-5 (%)   Throughput (imgs/sec)\n"
                + "".join(f"{gl(k):<12}  {t1:<9}  {t5:<9}  {p}\n"
                          for k, t1, t5, p in ROWS_A)
                + _A_CLOSE)
    # flat
    return (_A_CAPTION
            + "Kernel size Top-1 (%) Top-5 (%) Throughput (imgs/sec) "
            + " ".join(f"{gl(k)} {t1} {t5} {p}" for k, t1, t5, p in ROWS_A)
            + "\n" + _A_CLOSE)


TABLE_A_TEXT = _table_a_text(os.environ.get("TABLE_PROBE_STYLE", "markdown"))

# ── TABLE B: ablation path (order matters: before -> change -> after) ──
STEPS_B = [
    ("Swin-T baseline",                 "78.6"),
    ("+ overlapping tokenizer convs",   "79.4"),
    ("+ neighborhood attention",        "80.8"),
    ("+ larger 9x9 kernel",             "83.6"),
    ("NAT-Tiny (final)",                "84.2"),
]
B_VALS = {v for _, v in STEPS_B}
ALL_VALS = A_TOP1 | A_TOP5 | A_THR | B_VALS

_B_CAPTION = (
    "Table 6 shows the ablation path from the Swin-T baseline to NAT-Tiny. "
    "Each row adds one component on top of the previous row.\n\n"
)
_B_CLOSE = ("\nNeighborhood attention plus the larger kernel accounts for "
            "most of the improvement over the baseline.\n")

def _table_b_text(style):
    if style == "markdown":
        return (_B_CAPTION
                + "| Model variant | Top-1 (%) |\n"
                + "|---------------|-----------|\n"
                + "".join(f"| {n} | {v} |\n" for n, v in STEPS_B)
                + _B_CLOSE)
    if style == "aligned":
        return (_B_CAPTION
                + "Model variant                    Top-1 (%)\n"
                + "".join(f"{n:<32} {v}\n" for n, v in STEPS_B)
                + _B_CLOSE)
    return (_B_CAPTION
            + "Model variant Top-1 (%) "
            + " ".join(f"{n} {v}" for n, v in STEPS_B)
            + "\n" + _B_CLOSE)


TABLE_B_TEXT = _table_b_text(os.environ.get("TABLE_PROBE_STYLE", "markdown"))

SYSTEM_PART = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
ASSIST_PART = "<|im_end|>\n<|im_start|>assistant\n"

LIST_A_Q = ("Based on Table 4 above, reproduce the kernel size ablation "
            "exactly: for each kernel size, one line in the format "
            "'KxK: top1=NUMBER, top5=NUMBER, throughput=NUMBER imgs/sec'.")
LIST_B_Q = ("Based on Table 6 above, reproduce the ablation path exactly, in "
            "order, one line per row in the format 'variant: NUMBER'.")
SEQ_B_Q = ("According to Table 6 above, what was the Top-1 accuracy "
           "immediately after adding the overlapping tokenizer convolutions "
           "to the Swin-T baseline? Answer with only the number.")

def fwd_q(key):
    return (f"According to Table 4 above, what is the Top-1 accuracy of the "
            f"{key} kernel size? Answer with only the number.")

def thr_q(key):
    return (f"According to Table 4 above, what is the throughput in imgs/sec "
            f"of the {key} kernel size? Answer with only the number.")

def rev_q(val):
    return (f"According to Table 4 above, which kernel size achieves {val}% "
            f"Top-1 accuracy? Answer with only the kernel size.")

_NUM = re.compile(r"\d+(?:\.\d+)?")

def _key_pat(key):
    # (?<!\d)/(?!\d) instead of \b: models echo the question's 'KxK' format
    # as e.g. 'K3x3', and \b refuses the alpha-digit junction.
    a, b = key.split("x")
    return re.compile(rf"(?<!\d){a}\s*[x×]\s*{b}(?!\d)", re.IGNORECASE)

def score_list_a(text):
    """Segment by KEY POSITIONS, not lines: models sometimes flow all rows
    onto one line ('3x3 |79.1 |94.2 |1512 |5x5 |80.3 ...') — content-perfect
    but line-based scoring would call every row 'mixed'. Each key's segment
    runs from its match to the next key match (any planted key) or +60 chars."""
    res = {}
    clean = text.replace(",", "")
    matches = []            # (pos, key)
    for key, *_ in ROWS_A:
        m = _key_pat(key).search(clean)
        if m:
            matches.append((m.start(), key, m.end()))
    matches.sort()
    starts = [p for p, _, _ in matches]
    for key, top1, top5, thr in ROWS_A:
        hit = next((mm for mm in matches if mm[1] == key), None)
        if hit is None:
            res[key] = "miss"
            continue
        pos, _, end = hit
        nxt = min([p for p in starts if p > pos], default=end + 60)
        seg = clean[end:min(nxt, end + 60)]
        nums = set(_NUM.findall(seg))
        nums -= set(key.split("x"))
        own = {top1, top5, thr}
        hit_own = nums & own
        hit_other = nums & (ALL_VALS - own)
        if top1 in nums and not hit_other:
            verdict = "correct" if {top5, thr} <= nums else "partial"
        elif hit_other and hit_own:
            verdict = "mixed"
        elif hit_other:
            verdict = "swap"
        elif nums - own:
            verdict = "fab"
        elif hit_own:
            verdict = "partial"
        else:
            verdict = "miss"
        res[key] = verdict
    fab_rows = [k for k in FABRICATED_KEYS if _key_pat(k).search(clean)]
    return res, fab_rows

def score_list_b(text):
    """Per step: does a line mentioning the step's distinctive words carry its
    own value? Also check the VALUES appear in the planted order."""
    keywords = {
        "Swin-T baseline": ["swin"],
        "+ overlapping tokenizer convs": ["overlap"],
        "+ neighborhood attention": ["neighborhood", "neighbourhood"],
        "+ larger 9x9 kernel": ["larger", "9x9", "9×9"],
        "NAT-Tiny (final)": ["nat", "final"],
    }
    res = {}
    lines = [l.lower() for l in text.splitlines()]
    for name, val in STEPS_B:
        verdict = "miss"
        for ln in lines:
            if any(kw in ln for kw in keywords[name]):
                nums = set(_NUM.findall(ln.replace(",", "")))
                nums -= {"9", "1"}
                if val in nums:
                    verdict = "correct"
                elif nums & (ALL_VALS - {val}):
                    verdict = "swap"
                elif nums:
                    verdict = "fab"
                break
        res[name] = verdict
    seq = [n for n in _NUM.findall(text.replace(",", "")) if n in B_VALS]
    planted_order = [v for _, v in STEPS_B]
    order_ok = seq == planted_order
    return res, order_ok

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
    for key, *_ in ROWS_A:
        if key != expect and _key_pat(key).search(answer):
            return "swap"
    for key in FABRICATED_KEYS:
        if _key_pat(key).search(answer):
            return "fab"
    return "miss"

def build_prompt(tokenizer, ctx, question):
    """Real-paper filler; TABLE A starts 120 tokens BEFORE a block boundary so
    rows 3-4 straddle it; TABLE B sits fully inside a block in the second half."""
    with open(FILLER_FILE) as f:
        paper = f.read().replace("\x00", " ")
    sys_ids = tokenizer.encode(SYSTEM_PART, add_special_tokens=False)
    q_ids = tokenizer.encode("\n\n" + question + ASSIST_PART, add_special_tokens=False)
    ta_ids = tokenizer.encode("\n\n" + TABLE_A_TEXT + "\n", add_special_tokens=False)
    tb_ids = tokenizer.encode("\n\n" + TABLE_B_TEXT + "\n", add_special_tokens=False)
    filler_ids = tokenizer.encode(paper, add_special_tokens=False)
    body_budget = ctx - len(sys_ids) - len(q_ids) - len(ta_ids) - len(tb_ids)
    n_rep = body_budget // len(filler_ids) + 1
    filler_all = (filler_ids * n_rep)[:body_budget]

    # Table A: pick the block boundary nearest 40% depth; start A 120 tokens
    # before it (header ~90 tokens -> boundary cuts through the data rows).
    target_a = int(ctx * 0.4)
    boundary = (target_a // BLOCK) * BLOCK
    a_start = boundary - 120 - len(sys_ids)
    # Table B: fully inside a block near 70% depth: place so it doesn't cross
    # a boundary (table B is ~120 tokens; put it 8 tokens after a boundary).
    target_b = int(ctx * 0.7)
    b_start_abs = (target_b // BLOCK) * BLOCK + 8
    b_start = b_start_abs - len(sys_ids) - len(ta_ids)

    seg1 = filler_all[:a_start]
    seg2 = filler_all[a_start:b_start]
    seg3 = filler_all[b_start:]
    ids = sys_ids + seg1 + ta_ids + seg2 + tb_ids + seg3 + q_ids
    return tokenizer.decode(ids)

def run_cell(mode, ctx, question, max_tokens):
    import torch
    sys.path.insert(0, ACTIVE)
    os.chdir(ACTIVE)
    os.environ["DKV_COMPRESSED_DECODE"] = "0" if mode == "dense" else "1"
    os.environ.setdefault("DKV_LEGO_PREFILL", "0")
    from serving.hf_dkv_wrapper import DKVHFWrapper
    cfg = {"quantization": "int4", "rank": 16, "block_size": 256,
           "micro_block_size": 256, "preset": "mid"}
    wrapper = DKVHFWrapper(model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit", config=cfg)
    wrapper.ensure_loaded()
    tok, mgr, model = wrapper.tokenizer, wrapper.manager, wrapper.model
    prompt = build_prompt(tok, ctx, question)
    ids = tok.encode(prompt)
    sid = "table_probe2"
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

KINDS = {
    "listA": (LIST_A_Q, 280), "listB": (LIST_B_Q, 160), "seqB": (SEQ_B_Q, 16),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--modes", nargs="+", default=["dense", "dkv"])
    ap.add_argument("--full-matrix", action="store_true")
    ap.add_argument("--extra-env", nargs="*", default=[],
                    help="KEY=VAL pairs forwarded to cell subprocesses")
    ap.add_argument("--single", nargs=3, metavar=("MODE", "KIND", "ARG"))
    args = ap.parse_args()

    if args.single:
        mode, kind, arg = args.single
        if kind in KINDS:
            q, mt = KINDS[kind]
            run_cell(mode, args.ctx, q, mt)
        elif kind == "fwd":
            run_cell(mode, args.ctx, fwd_q(arg), 16)
        elif kind == "thr":
            run_cell(mode, args.ctx, thr_q(arg), 16)
        else:
            val = {k: t1 for k, t1, _, _ in ROWS_A}[arg]
            run_cell(mode, args.ctx, rev_q(val), 16)
        return

    env_extra = dict(kv.split("=", 1) for kv in args.extra_env)

    def spawn(mode, kind, arg):
        cmd = [sys.executable, os.path.abspath(__file__), "--ctx", str(args.ctx),
               "--single", mode, kind, arg]
        env = dict(os.environ, **env_extra)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env)
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("{"):
                try:
                    return json.loads(ln)["answer"]
                except Exception:
                    pass
        print(f"  [cell {mode}/{kind}/{arg} failed]\n{(r.stderr or '')[-500:]}", file=sys.stderr)
        return ""

    MATRIX_A = ["5x5", "7x7", "9x9", "11x11"]
    summary = {}
    for mode in args.modes:
        t0 = time.time()
        print(f"\n=== MODE {mode} ctx={args.ctx} extra={env_extra} ===", flush=True)
        ans = spawn(mode, "listA", "-")
        table, fab_rows = score_list_a(ans)
        ca = {k: sum(1 for v in table.values() if v == k)
              for k in ("correct", "partial", "mixed", "swap", "fab", "miss")}
        print(f"LIST-A [{mode}]: {ca}  fabricated_rows={fab_rows}")
        print(f"  per-row: {table}")
        print(f"  output: {ans[:500]!r}")
        ans_b = spawn(mode, "listB", "-")
        table_b, order_ok = score_list_b(ans_b)
        cb = {k: sum(1 for v in table_b.values() if v == k)
              for k in ("correct", "swap", "fab", "miss")}
        print(f"LIST-B [{mode}]: {cb}  order_ok={order_ok}")
        print(f"  per-step: {table_b}")
        print(f"  output: {ans_b[:400]!r}")
        a_seq = spawn(mode, "seqB", "-")
        v_seq = classify_num(a_seq, "79.4", ALL_VALS - {"79.4"})
        print(f"SEQ-B [{mode}]: {v_seq}  ({a_seq.strip()[:40]!r})")
        summary[mode] = {"listA": ca, "fab_rows": fab_rows, "listB": cb,
                         "order_ok": order_ok, "seqB": v_seq}
        if args.full_matrix:
            for kind in ("fwd", "thr", "rev"):
                res = []
                for key in MATRIX_A:
                    row = next(r for r in ROWS_A if r[0] == key)
                    a = spawn(mode, kind, key)
                    if kind == "fwd":
                        v = classify_num(a, row[1], ALL_VALS - {row[1]})
                    elif kind == "thr":
                        v = classify_num(a, row[3], ALL_VALS - {row[3]})
                    else:
                        v = classify_key(a, key)
                    res.append((key, v, a.strip()[:40]))
                cnt = {k: sum(1 for _, v, _ in res if v == k)
                       for k in ("correct", "swap", "fab", "miss")}
                print(f"{kind.upper()}-A [{mode}]: {cnt}")
                for r_ in res: print(f"   {r_}")
                summary[mode][kind] = cnt
        print(f"  mode wall time {time.time()-t0:.0f}s", flush=True)

    print("\n===== TABLE PROBE v2 SUMMARY =====")
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
