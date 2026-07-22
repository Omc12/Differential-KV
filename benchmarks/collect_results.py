#!/usr/bin/env python3
"""
collect_results.py — Collects all DKV eval JSON outputs and prints a
paper-ready summary table + identifies any missing / failed evals.

Usage:
    python3 benchmarks/collect_results.py
    python3 benchmarks/collect_results.py --json   # emit JSON to stdout
"""
import json
import os
import sys
import argparse

HERE  = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(HERE)
RES   = os.path.join(HERE, "results")

EXPECTED = {
    "test1_multi_needle.json":    "B1 Multi-needle NIAH (4 needles)",
    "test2_multihop.json":        "B2 Multi-hop NIAH",
    "test3_perplexity.json":      "B3 Perplexity",
    "test4_llama3b_niah.json":    "B4 Llama-3.2-3B NIAH",
    "test5_signal_ablation.json": "B5 Residual signal ablation",
    "test6_latency_breakdown.json": "B7 Latency breakdown (real)",
    "test7_lego_prefill_mem.json":  "B6 Lego prefill peak memory",
}

def load(fname):
    p = os.path.join(RES, fname)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}

def hr():
    print("-" * 72)

def section(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON summary to stdout")
    args = ap.parse_args()

    summary = {}

    # ── A1: Main NIAH Sweep ──────────────────────────────────────────────────
    section("A1 — Main NIAH Sweep (Active vs Dense @ 4k/8k/16k/32k/64k)")
    d_latest = load("results_latest.json")
    if d_latest is None or "_error" in d_latest:
        print("  [MISSING / RUNNING] Run: benchmarks/run_bench.py")
        summary["A1"] = "MISSING"
    else:
        results_list = d_latest.get("results", [])
        print(f"  {'Context':>8}  {'Engine':>8}  {'Status':>8}  {'Prefill(s)':>10}  {'tok/s':>8}  {'Needle':>7}")
        hr()
        for r in results_list:
            ctx = r.get("ctx_target", "?")
            eng = r.get("engine", "?")
            st  = r.get("status", "?")
            pf  = r.get("prefill_s", float("nan"))
            tps = r.get("decode_tps", float("nan"))
            nd  = "PASS" if r.get("needle_found") else "FAIL"
            print(f"  {ctx:>8}  {eng:>8}  {st:>8}  {pf:>10.2f}  {tps:>8.1f}  {nd:>7}")
        summary["A1"] = len(results_list)

    # ── C2: RULER ───────────────────────────────────────────────────────────
    section("C2 — RULER Benchmark")
    ruler_files = sorted(
        [f for f in os.listdir(RES) if f.startswith("ruler_results")],
        reverse=True,
    )
    if not ruler_files:
        print("  [MISSING] Run: benchmarks/run_ruler_mlx.py")
        summary["C2"] = "MISSING"
    else:
        d_ruler = load(ruler_files[0])
        if d_ruler and "_error" not in d_ruler:
            print(f"  File: {ruler_files[0]}")
            res_dict = d_ruler.get("results", {})
            contexts = d_ruler.get("contexts", [])
            header = f"  {'Task':<22}" + "".join(f"  {c//1024}k DKV  {c//1024}k Dense" for c in contexts)
            print(header)
            hr()
            for task_name, task_data in res_dict.items():
                row = f"  {task_name:<22}"
                for c in contexts:
                    cd = task_data.get(str(c)) or task_data.get(c) or {}
                    dk = cd.get("dkv", {}).get("accuracy", float("nan"))
                    dn = cd.get("dense", {}).get("accuracy", float("nan"))
                    row += f"  {dk:>8.1f}  {dn:>8.1f}"
                print(row)
            summary["C2"] = d_ruler

    # ── B1: Multi-needle ────────────────────────────────────────────────────
    section("B1 — Multi-needle NIAH (4 needles @ 4k/16k/32k)")
    d = load("test1_multi_needle.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_multi_needle_mlx.py")
        summary["B1"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B1"] = f"ERROR: {d['_error']}"
    else:
        print(f"  {'Context':>8}  {'Needles Found':>14}  {'Recall%':>8}  {'tok/s':>7}")
        hr()
        for r in d:
            ctx = r.get("context_tokens", "?")
            found = r.get("needles_found", "?")
            total = r.get("needles_total", 4)
            recall = r.get("recall_pct", 0.0)
            tps = r.get("tps", 0.0)
            print(f"  {ctx:>8}  {f'{found}/{total}':>14}  {recall:>7.1f}%  {tps:>7.1f}")
        summary["B1"] = [{"ctx": r["context_tokens"], "recall_pct": r["recall_pct"]} for r in d]

    # ── B2: Multi-hop ───────────────────────────────────────────────────────
    section("B2 — Multi-hop NIAH (chain recall @ 4k/16k/32k)")
    d = load("test2_multihop.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_multihop_mlx.py")
        summary["B2"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B2"] = f"ERROR: {d['_error']}"
    else:
        print(f"  {'Context':>8}  {'Pass/Fail':>10}  {'tok/s':>7}")
        hr()
        for r in d:
            ctx = r.get("context_tokens", "?")
            ok = "PASS" if r.get("success") else "FAIL"
            tps = r.get("tps", 0.0)
            print(f"  {ctx:>8}  {ok:>10}  {tps:>7.1f}")
        summary["B2"] = [{"ctx": r["context_tokens"], "pass": r["success"]} for r in d]

    # ── B3: Perplexity ───────────────────────────────────────────────────────
    section("B3 — Perplexity (dense vs DKV @ 4k/8k/16k)")
    d = load("test3_perplexity.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_ppl_mlx.py")
        summary["B3"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B3"] = f"ERROR: {d['_error']}"
    else:
        print(f"  {'Context':>8}  {'Dense PPL':>10}  {'DKV PPL':>12}  {'Delta%':>8}")
        hr()
        for r in d:
            ctx = r.get("context_tokens", "?")
            pd  = r.get("ppl_dense", float("nan"))
            pk  = r.get("ppl_dkv", float("nan"))
            dlt = r.get("ppl_delta_pct", float("nan"))
            print(f"  {ctx:>8}  {pd:>10.4f}  {pk:>12.4f}  {dlt:>+8.2f}%")
        summary["B3"] = [{"ctx": r["context_tokens"], "ppl_delta_pct": r["ppl_delta_pct"]} for r in d]

    # ── B4: Llama generalization ─────────────────────────────────────────────
    section("B4 — Llama-3.2-3B Cross-arch NIAH (@ 4k/8k/16k, depths 0.1/0.5/0.9)")
    d = load("test4_llama3b_niah.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_llama3b_mlx.py")
        summary["B4"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B4"] = f"ERROR: {d['_error']}"
    else:
        print(f"  {'Context':>8}  {'Depth':>6}  {'Pass/Fail':>10}  {'tok/s':>7}")
        hr()
        for r in d:
            ctx = r.get("context_tokens", "?")
            depth = r.get("depth", "?")
            ok = "PASS" if r.get("success") else "FAIL"
            tps = r.get("tps", 0.0)
            print(f"  {ctx:>8}  {depth:>6.1f}  {ok:>10}  {tps:>7.1f}")
        pass_n = sum(1 for r in d if r.get("success"))
        total_n = len(d)
        print(f"\n  Overall: {pass_n}/{total_n} ({100*pass_n/total_n:.0f}% recall)")
        summary["B4"] = {"pass": pass_n, "total": total_n}

    # ── B5: Signal ablation ──────────────────────────────────────────────────
    section("B5 — Residual Signal Ablation (@ 8k)")
    d = load("test5_signal_ablation.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_signal_ablation_mlx.py")
        summary["B5"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B5"] = f"ERROR: {d['_error']}"
    else:
        print(f"  {'Arm':<22}  {'Correct':>8}  {'Accuracy%':>10}  {'Swaps':>6}  {'Misses':>7}")
        hr()
        for r in d:
            arm = r.get("arm", "?")
            sc  = r.get("scores", {})
            acc = r.get("accuracy_pct", 0.0)
            print(f"  {arm:<22}  {sc.get('correct','?'):>4}/{sc.get('total','?'):<3}  {acc:>9.1f}%  {sc.get('swaps','?'):>6}  {sc.get('misses','?'):>7}")
        summary["B5"] = [{"arm": r["arm"], "accuracy_pct": r["accuracy_pct"]} for r in d]

    # ── B6: Lego prefill ─────────────────────────────────────────────────────
    section("B6 — Lego Streaming Prefill Peak Memory (16k/32k/48k)")
    d = load("test7_lego_prefill_mem.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_lego_mem_mlx.py")
        summary["B6"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B6"] = f"ERROR: {d['_error']}"
    else:
        print(f"  {'Context':>8}  {'Std VRAM GB':>12}  {'Lego VRAM GB':>13}  {'Saved GB':>9}  {'Reduction%':>11}")
        hr()
        for r in d:
            ctx   = r.get("context_tokens", "?")
            vstd  = r.get("standard_prefill_peak_vram_gb", "OOM")
            vlego = r.get("lego_prefill_peak_vram_gb", "OOM")
            saved = r.get("vram_saved_gb", 0.0)
            pct   = r.get("vram_reduction_pct", 0.0)
            print(f"  {ctx:>8}  {str(vstd):>12}  {str(vlego):>13}  {saved:>9.3f}  {pct:>10.1f}%")
        summary["B6"] = [{"ctx": r["context_tokens"], "reduction_pct": r["vram_reduction_pct"]} for r in d]

    # ── B7: Latency breakdown ────────────────────────────────────────────────
    section("B7 — Real Decode Latency Breakdown @ 16k")
    d = load("test6_latency_breakdown.json")
    if d is None:
        print("  [MISSING] Run: benchmarks/run_latency_breakdown_mlx.py --ctx 16000")
        summary["B7"] = "MISSING"
    elif "_error" in d:
        print(f"  [ERROR] {d['_error']}")
        summary["B7"] = f"ERROR: {d['_error']}"
    else:
        bd  = d.get("decode_step_breakdown", {})
        pct = d.get("breakdown_pct", {})
        total_ms = bd.get("total_step_ms", 0.0)
        print(f"  Total step: {total_ms:.1f} ms  ({bd.get('tok_per_sec', 0.0):.1f} tok/s)\n")
        print(f"  {'Component':<35}  {'ms':>6}  {'%':>6}")
        hr()
        rows = [
            ("Dense recency attention",          "dense_recency_attention_ms",    "dense_recency_attention"),
            ("Routing overhead (K=1 delta)",      "routing_overhead_ms",           "routing"),
            ("Low-rank scoring + residual attend","lowrank_scoring_residual_ms",   "lowrank_scoring_residual"),
            ("Fused buffer materialisation",      "buffer_materialisation_ms",     "buffer_materialisation"),
        ]
        for label, ms_key, pct_key in rows:
            ms_val  = bd.get(ms_key, 0.0)
            pct_val = pct.get(pct_key, 0.0)
            print(f"  {label:<35}  {ms_val:>6.1f}  {pct_val:>5.1f}%")
        print(f"\n  Note: {d.get('note','')[:120]}...")
        summary["B7"] = bd

    # ── Check for LongBench results ──────────────────────────────────────────
    section("C1 — LongBench (NarrativeQA / Qasper / HotpotQA / GovReport)")
    # LongBench output is timestamped, find the latest
    lb_files = sorted(
        [f for f in os.listdir(RES) if f.startswith("longbench_compare")],
        reverse=True,
    )
    if not lb_files:
        print("  [MISSING] Run: ACTIVE_RUNTIME/run_longbench.py --compare")
        summary["C1"] = "MISSING"
    else:
        lb_path = os.path.join(RES, lb_files[0])
        try:
            with open(lb_path) as f:
                lb = json.load(f)
            print(f"  File: {lb_files[0]}")
            print(f"  {'Dataset':<16}  {'Metric':>7}  {'Dense':>8}  {'DKV':>8}  {'Delta':>8}")
            hr()
            for dataset, ddata in lb.items():
                if isinstance(ddata, dict) and "dense" in ddata and "dkv" in ddata:
                    metric = ddata.get("metric", "?")
                    dv = ddata["dense"].get(metric, float("nan"))
                    kv = ddata["dkv"].get(metric, float("nan"))
                    delta = kv - dv if isinstance(dv, float) else float("nan")
                    print(f"  {dataset:<16}  {metric:>7}  {dv:>8.3f}  {kv:>8.3f}  {delta:>+8.3f}")
            summary["C1"] = lb
        except Exception as e:
            print(f"  [ERROR parsing {lb_files[0]}]: {e}")
            summary["C1"] = f"ERROR: {e}"

    # ── Overall status ───────────────────────────────────────────────────────
    section("Overall Eval Status")
    missing = [k for k, v in summary.items() if v == "MISSING"]
    errored = [k for k, v in summary.items() if isinstance(v, str) and v.startswith("ERROR")]
    done    = [k for k in summary if k not in missing and k not in errored]
    print(f"  Done   : {', '.join(done) if done else 'none'}")
    print(f"  Missing: {', '.join(missing) if missing else 'none'}")
    print(f"  Errors : {', '.join(errored) if errored else 'none'}")
    print()

    if args.json:
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
