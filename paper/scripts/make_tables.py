#!/usr/bin/env python3
"""Emit LaTeX tables straight from measured JSON (no hand-typed numbers).
Outputs -> paper/tables/*.tex
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TBL = os.path.join(HERE, "..", "tables")
os.makedirs(TBL, exist_ok=True)
A_JSON = os.path.join(REPO, "paper/generated/active_modes_sweep.json")
D_JSON = os.path.join(REPO, "benchmarks/results/PAPER_dense_sweep.json")
CTXS = [4096, 8192, 16384, 32768, 65536]


def load_active():
    if not os.path.exists(A_JSON): return {}
    out = {"compressed": {}, "exact": {}}
    for r in json.load(open(A_JSON))["results"]:
        if "decode_tps" in r:
            out.setdefault(r["mode"], {})[r["ctx"]] = r
    return out


def load_dense():
    if not os.path.exists(D_JSON): return {}
    return {r["ctx_target"]: r for r in json.load(open(D_JSON))["results"] if r.get("status") == "ok"}


def f(v, s="%.1f"):
    return s % v if isinstance(v, (int, float)) else "--"


def write(name, body):
    with open(os.path.join(TBL, name), "w") as fp:
        fp.write(body)
    print("wrote", name)


def t_main(A, D):
    """T3: main results — compressed (primary), exact (ablation), dense (baseline)."""
    comp, exa = A.get("compressed", {}), A.get("exact", {})
    rows = []
    hdr = " & ".join(["%dk" % (c // 1024) for c in CTXS])
    def line(label, dct, key, sub=None, s="%.1f", scale=1.0):
        cells = []
        for c in CTXS:
            r = dct.get(c)
            if r is None: cells.append("--"); continue
            v = r[sub][key] if sub else r.get(key)
            if v is not None and isinstance(v, (int, float)) and scale != 1.0:
                v = v * scale
            cells.append(f(v, s) if v is not None else "--")
        return label + " & " + " & ".join(cells) + r" \\"
    L = []
    L.append(r"\begin{tabular}{l" + "r"*len(CTXS) + "}")
    L.append(r"\toprule")
    L.append(r"Metric & " + hdr + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{DiffKV compressed sparse decode (primary)}}\\" % (len(CTXS)+1))
    L.append(line("\\quad Prefill (s)", comp, "prefill_s", s="%.1f"))
    L.append(line("\\quad Decode (tok/s)", comp, "decode_tps", s="%.1f"))
    L.append(line("\\quad Decode-phase MLX peak (GB)", comp, "mx_decode_peak_gb", s="%.2f"))
    L.append(line("\\quad KV store, occupied (GB)", comp, "store_used_bytes", sub="kv", s="%.3f", scale=1e-9))
    L.append(line("\\quad Needle recovered", comp, "needle_found", s="%s"))
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Exact full-KV decode (upper-bound ablation)}}\\" % (len(CTXS)+1))
    L.append(line("\\quad Decode (tok/s)", exa, "decode_tps", s="%.1f"))
    L.append(line("\\quad Needle recovered", exa, "needle_found", s="%s"))
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\textit{Dense baseline (mlx\_lm, full KV)}}\\" % (len(CTXS)+1))
    dd = {c: D.get(c) for c in CTXS}
    L.append(line("\\quad Prefill (s)", dd, "prefill_s", s="%.1f"))
    L.append(line("\\quad Decode (tok/s)", dd, "decode_tps", s="%.1f"))
    L.append(line("\\quad Full KV-cache (GB, analytic)", comp, "dense_full_bytes", sub="kv", s="%.3f", scale=1e-9))
    L.append(line("\\quad Needle recovered", dd, "needle_found", s="%s"))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    body = "\n".join(L).replace("True", r"\cmark").replace("False", r"\xmark").replace("None", "--")
    write("t3_main_results.tex", body)


def t_config():
    """T1 + T2 are static (from code) — emitted by the paper directly; here we emit
    the derived per-block budget so it stays in sync if dims change."""
    Hkv, d, B, r, M, Dmax, L = 2, 128, 256, 16, 256, 768, 28
    per_block = (B-1)*r*2 + 2*Hkv*r*d*2 + 2*Hkv*d*2 + 8
    dense_block = B*Hkv*d*2*2
    lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Component & DiffKV block & Dense block \\", r"\midrule",
        r"U coefficients $[255,16]$ fp16 & %d B & -- \\" % ((B-1)*r*2),
        r"$V_K,V_V\ [2,16,128]$ fp16 & %d B & -- \\" % (2*Hkv*r*d*2),
        r"anchors $a_k,a_v\ [2,128]$ fp16 & %d B & -- \\" % (2*Hkv*d*2),
        r"scale + seq\_len & 8 B & -- \\",
        r"full $K,V\ [256,2,128]$ fp16 & -- & %d B \\" % dense_block,
        r"\midrule",
        r"\textbf{Per 256-token block} & \textbf{%s KiB} & \textbf{%s KiB} \\" % (
            "%.1f" % (per_block/1024), "%.0f" % (dense_block/1024)),
        r"Compression ratio & \multicolumn{2}{c}{$%.2f\times$} \\" % (dense_block/per_block),
        r"\bottomrule", r"\end{tabular}",
    ]
    write("t2_block_budget.tex", "\n".join(lines))


def t_detail(A, D):
    """T5 appendix: full per-run detail, every measured metric."""
    L = [r"\begin{tabular}{llrrrrrrl}", r"\toprule",
         r"Config & Ctx & Prefill (s) & Decode tok/s & MLX peak (GB) & Dec.\ peak (GB) & "
         r"Store (GB) & Full KV (GB) & Needle \\", r"\midrule"]
    def rows(dct, label):
        for c in CTXS:
            r = dct.get(c)
            if not r: continue
            k = r.get("kv", {})
            nd = r"\cmark" if r.get("needle_found") else r"\xmark"
            L.append("%s & %dk & %s & %s & %s & %s & %s & %s & %s \\\\" % (
                label, c//1024, f(r.get("prefill_s")), f(r.get("decode_tps")),
                f(r.get("mx_peak_gb"), "%.2f"), f(r.get("mx_decode_peak_gb"), "%.2f"),
                f(k.get("store_used_bytes", 0)/1e9 if k else None, "%.3f"),
                f(k.get("dense_full_bytes", 0)/1e9 if k else None, "%.3f"), nd))
    rows(A.get("compressed", {}), "compressed")
    L.append(r"\midrule")
    rows(A.get("exact", {}), "exact")
    L.append(r"\midrule")
    # dense
    for c in CTXS:
        r = D.get(c)
        if not r: continue
        nd = r"\cmark" if r.get("needle_found") else r"\xmark"
        L.append("dense & %dk & %s & %s & %s & -- & -- & -- & %s \\\\" % (
            c//1024, f(r.get("prefill_s")), f(r.get("decode_tps")),
            f(r.get("mx_peak_gb"), "%.2f"), nd))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t5_detail.tex", "\n".join(L))


if __name__ == "__main__":
    A, D = load_active(), load_dense()
    t_main(A, D)
    t_config()
    t_detail(A, D)
    print("tables done")
