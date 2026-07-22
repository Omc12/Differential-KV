#!/usr/bin/env python3
"""Emit every LaTeX table for the paper straight from data.py (measured JSON or
code-derived dimensions). No hand-typed numbers. Outputs -> paper/tables/*.tex

  T1  Model + DKV configuration (code-derived)
  T2  Per-block byte budget & compression ratio (code-derived; R=128 & R=64)
  T3  Main results — DKV (active) vs dense (fresh primary sweep)
  T4  Residual-budget accuracy/memory sweep (measured)
  T5  Per-run detail (appendix; fresh primary)
  T6  Compressed-vs-exact decode ablation (measured modes)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

TBL = os.path.join(os.path.dirname(HERE), "tables")
os.makedirs(TBL, exist_ok=True)


def write(name, body):
    with open(os.path.join(TBL, name), "w") as fp:
        fp.write(body + "\n")
    print("wrote", name)


def _cm(b):
    return r"\cmark" if b else r"\xmark"


def _f(v, s="%.1f"):
    return s % v if isinstance(v, (int, float)) else "--"


# ── T1 · model + configuration ───────────────────────────────────────────────
def t1_config():
    d = D.DIMS
    rows = [
        ("Model", d["model"]),
        ("Transformer layers $L$", f"{d['n_layers']}"),
        ("Query / KV heads (GQA)", f"{d['n_heads']} / {d['kv_heads']}"),
        ("Head dimension $d$", f"{d['head_dim']}"),
        ("Block size $B$", f"{d['block_size']} tokens"),
        ("SVD rank $r$", f"{d['rank']} (adaptive $\\le r$, 99.9\\% energy)"),
        ("Recency window $W$ (+block)", f"{d['recency_window']} (+{d['block_size']} = 768 exact)"),
        ("Residual budget $R$", "128 (default) / 64 (memory preset)"),
        ("Block pool $M$", f"{d['max_blocks']} blocks ($\\le$ 65\\,536 tokens)"),
        ("Top-$K$ routed blocks", "16 (residual-key router)"),
        ("Store dtype", "fp16"),
        ("SVD seed", "1234 (deterministic)"),
    ]
    L = [r"\begin{tabular}{ll}", r"\toprule",
         r"Parameter & Value \\", r"\midrule"]
    L += [f"{k} & {v} \\\\" for k, v in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t1_config.tex", "\n".join(L))


# ── T2 · per-block byte budget ───────────────────────────────────────────────
def t2_block_budget():
    b128 = D.block_budget(128)
    b64 = D.block_budget(64)
    def kib(x):
        return "%.1f" % (x / 1024)
    L = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Component & Bytes & Note \\", r"\midrule",
         r"$U$ coefficients $[255,16]$ & %d & low-rank \\" % b128["U"],
         r"$V_K,V_V\ [2,16,128]$ & %d & low-rank \\" % b128["VKV"],
         r"anchors $a_k,a_v\ [2,128]$ & %d & exact \\" % b128["anchors"],
         r"key min/max $[2,128]$ & %d & router \\" % b128["minmax"],
         r"scale, seq\_len & %d & scalar \\" % b128["scalars"],
         r"\midrule",
         r"\textbf{Low-rank core} & \textbf{%d} & (%s KiB) \\" % (b128["lowrank"], kib(b128["lowrank"])),
         r"exact residuals $R{=}128$ & %d & (%s KiB) \\" % (b128["residuals"], kib(b128["residuals"])),
         r"exact residuals $R{=}64$ & %d & (%s KiB) \\" % (b64["residuals"], kib(b64["residuals"])),
         r"\midrule",
         r"\textbf{DKV block} ($R{=}128$) & \textbf{%d} & \textbf{%s KiB} \\" % (b128["total"], kib(b128["total"])),
         r"\textbf{DKV block} ($R{=}64$) & \textbf{%d} & \textbf{%s KiB} \\" % (b64["total"], kib(b64["total"])),
         r"Dense block $[256,2,128]{\times}2$ & %d & %s KiB \\" % (b128["dense"], kib(b128["dense"])),
         r"\midrule",
         r"Compression ratio ($R{=}128$) & \multicolumn{2}{r}{$%.2f\times$} \\" % b128["ratio"],
         r"Compression ratio ($R{=}64$) & \multicolumn{2}{r}{$%.2f\times$} \\" % b64["ratio"],
         r"\bottomrule", r"\end{tabular}"]
    write("t2_block_budget.tex", "\n".join(L))


# ── T3 · main results (fresh primary) ────────────────────────────────────────
def t3_main():
    prim = D.load_primary()
    # Include every context the active engine reached (64k included); dense may OOM.
    ctx = [c for c in D.CONTEXTS if c in prim["active"]]
    hdr = " & ".join("%dk" % (c // 1024) for c in ctx)

    def row(label, engine, key, s="%.1f"):
        cells = []
        for c in ctx:
            if c not in prim[engine]:
                st = D.cell_status(engine, c)
                cells.append(r"\textit{OOM}" if st == "oom" else "--")
                continue
            v = prim[engine][c].get(key)
            cells.append(_f(v, s) if not isinstance(v, bool) else _cm(v))
        return label + " & " + " & ".join(cells) + r" \\"

    L = [r"\begin{tabular}{l" + "r" * len(ctx) + "}", r"\toprule",
         r"Metric & " + hdr + r" \\", r"\midrule",
         r"\multicolumn{%d}{l}{\textit{DKV active runtime (compressed sparse decode)}}\\" % (len(ctx) + 1),
         row(r"\quad Prefill (s)", "active", "prefill_s"),
         row(r"\quad Decode (tok/s)", "active", "decode_tps"),
         row(r"\quad KV cache footprint (GB)", "active", "kv_mem_gb", "%.2f"),
         row(r"\quad Peak process memory (GB)", "active", "mx_peak_gb", "%.2f"),
         row(r"\quad Needle recovered", "active", "needle_found"),
         r"\midrule",
         r"\multicolumn{%d}{l}{\textit{Optimized dense baseline (mlx\_lm, full KV)}}\\" % (len(ctx) + 1),
         row(r"\quad Prefill (s)", "dense", "prefill_s"),
         row(r"\quad Decode (tok/s)", "dense", "decode_tps"),
         row(r"\quad KV cache footprint (GB)", "dense", "kv_mem_gb", "%.2f"),
         row(r"\quad Peak process memory (GB)", "dense", "mx_peak_gb", "%.2f"),
         row(r"\quad Needle recovered", "dense", "needle_found"),
         r"\midrule",
         r"\multicolumn{%d}{l}{\textit{Standard PyTorch dense baseline (AutoModelForCausalLM, full KV)}}\\" % (len(ctx) + 1),
         row(r"\quad Prefill (s)", "normal_dense", "prefill_s"),
         row(r"\quad Decode (tok/s)", "normal_dense", "decode_tps"),
         row(r"\quad KV cache footprint (GB)", "normal_dense", "kv_mem_gb", "%.2f"),
         row(r"\quad Peak process memory (GB)", "normal_dense", "mx_peak_gb", "%.2f"),
         row(r"\quad Needle recovered", "normal_dense", "needle_found"),
         r"\bottomrule", r"\end{tabular}"]
    write("t3_main_results.tex", "\n".join(L))


# ── T4 · residual sweep ──────────────────────────────────────────────────────
def t4_residual():
    rows = D.load_residual_sweep()
    if not rows:
        print("skip t4: no residual sweep"); return
    L = [r"\begin{tabular}{rccrr}", r"\toprule",
         r"$R$ & Needle & Decode (tok/s) & Store (GB) & Ratio vs dense \\", r"\midrule"]
    for r in rows:
        k = r["kv"]
        L.append("%d & %s & %s & %s & $%.2f\\times$ \\\\" % (
            r["max_residual"], _cm(r.get("needle_found")), _f(r.get("decode_tps")),
            _f(k["store_used_bytes"] / 1e9, "%.3f"), k["ratio_used_vs_dense"]))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("t4_residual.tex", "\n".join(L))


# ── T5 · per-run detail (appendix) ───────────────────────────────────────────
def t5_detail():
    prim = D.load_primary()
    L = [r"\begin{tabular}{llrrrrl}", r"\toprule",
         r"Engine & Ctx & Prompt tok & Prefill (s) & Decode (tok/s) & MLX peak (GB) & Needle \\",
         r"\midrule"]
    for engine in ("active", "dense"):
        for c in D.CONTEXTS:
            r = prim[engine].get(c)
            if not r:
                continue
            L.append("%s & %dk & %s & %s & %s & %s & %s \\\\" % (
                engine, c // 1024, _f(r.get("prompt_tokens"), "%d"),
                _f(r.get("prefill_s")), _f(r.get("decode_tps")),
                _f(r.get("mx_peak_gb"), "%.2f"), _cm(r.get("needle_found"))))
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    write("t5_detail.tex", "\n".join(L))


# ── T6 · compressed-vs-exact decode ablation ─────────────────────────────────
def t6_ablation():
    comp = D.modes_by("compressed")
    exact = D.modes_by("exact")
    ctx = [c for c in D.CONTEXTS if c in comp and c in exact]
    if not ctx:
        print("skip t6: no modes"); return
    hdr = " & ".join("%dk" % (c // 1024) for c in ctx)

    def row(label, dct, key, s="%.1f"):
        cells = [(_cm(dct[c].get(key)) if isinstance(dct[c].get(key), bool)
                  else _f(dct[c].get(key), s)) for c in ctx]
        return label + " & " + " & ".join(cells) + r" \\"

    L = [r"\begin{tabular}{l" + "r" * len(ctx) + "}", r"\toprule",
         r"Decode over the same DKV store & " + hdr + r" \\", r"\midrule",
         row(r"Compressed sparse decode (tok/s)", comp, "decode_tps"),
         row(r"Exact decode, upper bound (tok/s)", exact, "decode_tps"),
         r"\midrule",
         row(r"Compressed — needle", comp, "needle_found"),
         row(r"Exact — needle", exact, "needle_found"),
         r"\bottomrule", r"\end{tabular}"]
    write("t6_decode_ablation.tex", "\n".join(L))


if __name__ == "__main__":
    t1_config()
    t2_block_budget()
    t3_main()
    t4_residual()
    t5_detail()
    t6_ablation()
    print("tables done")
