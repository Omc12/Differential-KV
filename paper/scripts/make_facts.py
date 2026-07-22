#!/usr/bin/env python3
"""Emit paper/generated/facts.tex — every measured/derived number the prose cites,
as LaTeX macros. The body text uses these macros, so it can NEVER drift from the
measured data: regenerate this after any re-measurement and recompile.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "generated", "facts.tex")


def num(x, s="%.1f"):
    return s % x


def main():
    prim = D.load_primary()
    M = []

    def mac(name, val):
        M.append(r"\newcommand{\%s}{%s}" % (name, val))

    words = {4096: "FourK", 8192: "EightK", 16384: "SixteenK",
             32768: "ThirtyTwoK", 65536: "SixtyFourK"}
    # per-context active values (include 64k reach); dense only where it did not OOM
    for c in D.CONTEXTS:
        w = words[c]
        if c in prim["active"] and prim["active"][c].get("status") == "ok":
            a = prim["active"][c]
            mac(f"actPf{w}", num(a["prefill_s"], "%.0f" if c >= 65536 else "%.1f"))
            mac(f"actTps{w}", num(a["decode_tps"]))
            mac(f"actMx{w}", num(a.get("peak_mem_gb", a.get("mx_peak_gb", 0)), "%.2f"))
        else:
            mac(f"actPf{w}", "--")
            mac(f"actTps{w}", "--")
            mac(f"actMx{w}", "--")

        if c in prim["dense"] and prim["dense"][c].get("status") == "ok":
            d = prim["dense"][c]
            mac(f"dnPf{w}", num(d["prefill_s"], "%.0f" if c >= 65536 else "%.1f"))
            mac(f"dnTps{w}", num(d["decode_tps"]))
            mac(f"dnMx{w}", num(d.get("peak_mem_gb", d.get("mx_peak_gb", 0)), "%.2f"))
        else:
            mac(f"dnPf{w}", "--")
            mac(f"dnTps{w}", "--")
            mac(f"dnMx{w}", "--")

    both = [c for c in D.CONTEXTS if c in prim["active"] and c in prim["dense"] and prim["active"][c].get("status") == "ok" and prim["dense"][c].get("status") == "ok"]
    if 32768 in both:
        a, d = prim["active"][32768], prim["dense"][32768]
        mac("pfSpeedupThirtyTwoK", num(d["prefill_s"] / a["prefill_s"], "%.2f"))
        mac("decRatioThirtyTwoK", num(d["decode_tps"] / a["decode_tps"], "%.1f"))
    else:
        mac("pfSpeedupThirtyTwoK", "--")
        mac("decRatioThirtyTwoK", "--")

    if 4096 in both:
        a, d = prim["active"][4096], prim["dense"][4096]
        mac("decRatioFourK", num(d["decode_tps"] / a["decode_tps"], "%.1f"))
    else:
        mac("decRatioFourK", "--")

    # DKV prefill speedup over the optimized dense baseline at 64k
    if 65536 in both:
        a, d = prim["active"][65536], prim["dense"][65536]
        mac("pfSpeedupSixtyFourK", num(d["prefill_s"] / a["prefill_s"], "%.2f"))
    else:
        mac("pfSpeedupSixtyFourK", "--")

    # dense OOM at 64k? (optimized mlx_lm dense — now completes, so '--')
    mac("denseSixtyFourK", "OOM" if D.cell_status("dense", 65536) in ("oom", "error") else "--")

    # Standard PyTorch dense (AutoModelForCausalLM, naive full KV): reach boundary.
    # This naive baseline is the one that OOMs early on the 8.6 GB host.
    nd = prim["normal_dense"]
    nd_ok = [c for c in D.CONTEXTS if c in nd and nd[c].get("status") == "ok"]
    pt_oom = [c for c in D.CONTEXTS if D.cell_status("normal_dense", c) in ("oom", "error")]
    mac("ptMaxK", f"{max(nd_ok)//1024}" if nd_ok else "--")   # largest completed
    mac("ptOOMK", f"{min(pt_oom)//1024}" if pt_oom else "--")  # first OOM
    if nd_ok:
        lo = min(nd_ok)
        mac("ptTps", num(nd[lo]["decode_tps"]))                # ~3.6 tok/s even where it fits
        mac("ptMx", num(nd[lo].get("peak_mem_gb", nd[lo].get("mx_peak_gb", 0)), "%.2f"))
    else:
        mac("ptTps", "--"); mac("ptMx", "--")
    # reach = largest context with active needle recovered
    act = prim["active"]
    reach = max((c for c in act if act[c].get("needle_found")), default=max(act or [0]))
    mac("reachK", f"{reach//1024}")
    mac("maxCtxK", f"{max(both)//1024}" if both else "8")
    mac("nContexts", f"{len(D.active_contexts())}")

    # block budget
    b128, b64 = D.block_budget(128), D.block_budget(64)
    mac("blkTotalDefault", num(b128["total"] / 1024))
    mac("blkTotalPreset", num(b64["total"] / 1024))
    mac("blkDense", "%.0f" % (b128["dense"] / 1024))
    mac("blkLowrank", num(b128["lowrank"] / 1024))
    mac("ratioDefault", num(b128["ratio"], "%.2f"))
    mac("ratioPreset", num(b64["ratio"], "%.2f"))

    # analytic footprint @64k (whole model)
    for seq, w in [(65536, "SixtyFourK"), (32768, "ThirtyTwoK")]:
        f128 = D.analytic_footprint(seq, 128)
        f64 = D.analytic_footprint(seq, 64)
        mac(f"kvStore{w}Default", num(f128["store_bytes"] / 1e9, "%.2f"))
        mac(f"kvStore{w}Preset", num(f64["store_bytes"] / 1e9, "%.2f"))
        mac(f"kvDense{w}", num(f128["dense_bytes"] / 1e9, "%.2f"))
        mac(f"kvRatio{w}Default", num(f128["ratio"], "%.2f"))
        mac(f"kvRatio{w}Preset", num(f64["ratio"], "%.2f"))

    # residual sweep: recall threshold + endpoints
    rs = D.load_residual_sweep()
    if rs:
        first_ok = next((r for r in rs if r.get("needle_found")), None)
        if first_ok:
            mac("resThreshold", f"{first_ok['max_residual']}")
        lo, hi = rs[0], rs[-1]
        mac("resRatioLo", num(lo["kv"]["ratio_used_vs_dense"], "%.2f"))
        mac("resRatioHi", num(hi["kv"]["ratio_used_vs_dense"], "%.2f"))
        mac("resLo", f"{lo['max_residual']}")
        mac("resHi", f"{hi['max_residual']}")

    # compressed-vs-exact decode ablation @16k
    comp = D.modes_by("compressed"); exact = D.modes_by("exact")
    if 16384 in comp and 16384 in exact:
        mac("ablCompSixteenK", num(comp[16384]["decode_tps"]))
        mac("ablExactSixteenK", num(exact[16384]["decode_tps"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("% AUTO-GENERATED by paper/scripts/make_facts.py — do not edit.\n")
        f.write("\n".join(M) + "\n")
    print(f"wrote {OUT}  ({len(M)} macros)")
    print("active contexts:", [f"{c//1024}k" for c in D.active_contexts()])
    print("dense 64k status:", D.cell_status("dense", 65536))


if __name__ == "__main__":
    main()
