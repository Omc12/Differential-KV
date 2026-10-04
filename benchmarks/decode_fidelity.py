#!/usr/bin/env python3
"""Quick, decode-time fidelity of a DKV configuration against dense.

WHY NOT A WHOLE-PROMPT FORWARD
------------------------------
Teacher-forcing a prompt measures prefill, and DKV's exact prefill is dense's
(see bench_ppl_cuda.py). What compression changes is DECODE: every generated
token attends the compressed store. So each prompt is prefilled, and then
dense's own greedy answer is fed back token by token through DKV's decode:
an lm_head hook records DKV's distribution at each step and then forces
dense's token, so both arms condition on the same prefix at every step and
every step is a fair comparison (logit_fidelity.py scores only the first one,
because it let the arms diverge).

WHAT IT REPORTS, per arm (decode steps only; step 0 is the prefill's output
and is reported apart):
    kl        mean KL(dense || arm) per step
    top1      share of steps where the arm's argmax is dense's token
    dnll      mean NLL of dense's token under the arm minus under dense
and the guardrails a fix must not trade away:
    kv_gb / cmp   stored bytes and compression against a dense cache
    peak_gb       peak allocated memory over the prompt
    dec_tps       decode tokens per second

The default prompt set is where DKV loses: LongBench qasper and multifieldqa
at 12k, and RULER's lookalike-needle multi-key tasks and qa_1 at 16k. On
granite every one fits dense, and at these lengths routing attends every
block, so the arms measure compression alone.

USAGE
    python benchmarks/decode_fidelity.py --model ibm-granite/granite-4.2-8b \\
        --arms base "res256:DKV_MAX_RESIDUAL_TOKENS=256" "stream:DKV_STREAMING_COMPRESS=1"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, HERE)

# The fixed tier-2 set (shared with tier1_capture.py): every prompt is long
# enough that DKV compresses most of it.
LB_DEFAULT = ["qasper:15-15", "multifieldqa_en:3-3", "hotpotqa:1-1"]
RULER_DEFAULT = ["16384/niah_multikey_2:0-0", "16384/niah_multikey_3:0-0",
                 "16384/qa_1:0-0"]


def _span(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return range(int(a), int(b) + 1)


def build_items(tok, model_id, lb_specs, ruler_specs, max_length):
    """[(key, prompt)] for the requested LongBench and RULER items."""
    from run_longbench_cuda import official_config, load_task, build_prompt
    from run_ruler_cuda import load_generated, build_ruler_prompt
    items = []
    prompts, _ = official_config()
    for spec in lb_specs:
        ds, rng = spec.split(":")
        rows = load_task(ds, max(_span(rng)) + 1)
        for i in _span(rng):
            items.append(("lb/%s#%d" % (ds, i),
                          build_prompt(tok, rows[i], ds, prompts[ds], max_length)))
    if ruler_specs:
        tag = model_id.split("/")[-1]
        gen = load_generated(os.path.join(REPO, "paper", "results", "ruler_data", tag))
        for spec in ruler_specs:
            lt, rng = spec.split(":")
            L, task = lt.split("/")
            pool = {it["idx"]: it for it in gen
                    if it["length"] == int(L) and it["task"] == task}
            for i in _span(rng):
                items.append(("ruler/%s/%s#%d" % (L, task, i),
                              build_ruler_prompt(tok, pool[i])))
    return items


# ───────────────────────────────────────────────────────────── dense ──
def run_dense(args, out_path):
    import torch
    import kv_baselines as KB
    from run_longbench_cuda import load_plain, derive_stop_ids
    tok, model = load_plain(args.model, args.quant, False)
    stop = derive_stop_ids(tok)
    items = build_items(tok, args.model, args.lb, args.ruler, args.max_length)
    rec = {}
    for key, prompt in items:
        ids = tok(prompt).input_ids
        past, logits = KB.chunked_prefill(model, ids, "cuda", 1024)
        toks, lps = [], []
        cur = len(ids)
        with torch.no_grad():
            for _ in range(args.gen):
                lp = torch.log_softmax(logits.float(), -1)
                t = int(lp.argmax())
                toks.append(t)
                lps.append(lp.half().cpu())
                if t in stop:
                    break
                out = model(input_ids=torch.tensor([[t]], device="cuda"),
                            position_ids=torch.tensor([[cur]], device="cuda"),
                            past_key_values=past, use_cache=True)
                past, logits = out.past_key_values, out.logits[0, -1]
                cur += 1
        rec[key] = {"ntok": len(ids), "tokens": toks, "lp": torch.stack(lps)}
        print("  dense %s: %d prompt tokens, %d answer tokens" % (key, len(ids), len(toks)),
              flush=True)
        del past
        torch.cuda.empty_cache()
    torch.save(rec, out_path)


# ─────────────────────────────────────────────────────────────── dkv ──
def run_dkv(args, dense_path, out_path):
    import torch
    rec = torch.load(dense_path)
    os.environ.setdefault("DKV_RSVD_SEED", "1234")
    os.environ.setdefault("DKV_SVD_SEED", "1234")
    cwd = os.getcwd()
    os.chdir(ACTIVE)
    sys.path.insert(0, ACTIVE)
    from msvc_env import ensure_msvc
    ensure_msvc()
    # The lm_head hook runs host-side code inside the decode forward, which a
    # CUDA-graph capture cannot record: the capture fails and replays hand back
    # stale buffers (seen as KL 4-15 on exactly the prompts under the graph
    # context limit). Graphs change speed, not arithmetic, so every arm runs
    # without them; decode tokens/s here is therefore graph-free.
    os.environ["DKV_DISABLE_CUDA_GRAPH"] = "1"
    from serving.decode_config import apply_best_decode_defaults
    apply_best_decode_defaults()                 # the shipped decode path
    from serving.hf_dkv_wrapper import DKVHFWrapper
    from run_longbench_cuda import dkv_kv_bytes
    w = DKVHFWrapper(model_id=args.model,
                     config={"preset": args.preset, "quantization": args.quant or None})
    w.ensure_loaded()
    os.chdir(cwd)
    tok, model = w.tokenizer, w.model
    items = build_items(tok, args.model, args.lb, args.ruler, args.max_length)

    st = {"L": 0, "step": -1, "forced": [], "lp": [], "t": []}

    def _pre(_m, a, kw):
        ii = kw.get("input_ids")
        if ii is None and a:
            ii = a[0]
        st["L"] = int(ii.shape[1]) if ii is not None and hasattr(ii, "shape") else 0

    def _hook(_m, _inp, out):
        # Prefill chunks (L > 1) all produce "step 0"; the last one is the one
        # generation samples from. Decode steps (L == 1) are steps 1, 2, ...
        st["step"] = 0 if st["L"] > 1 else st["step"] + 1
        j = st["step"]
        forced = st["forced"]
        if j >= len(forced):
            return out
        lp = torch.log_softmax(out[0, -1].float(), -1)
        if j == 0:
            st["lp"][:1] = [lp]
        else:
            st["lp"].append(lp)
            st["t"].append(time.perf_counter())
        o = out.clone()
        o[0, -1] = o[0, -1].min()
        o[0, -1, forced[j]] = o[0, -1].max() + 1e4 if o[0, -1].max() > 0 else 1e4
        return o

    hp = model.register_forward_pre_hook(_pre, with_kwargs=True)
    hh = model.lm_head.register_forward_hook(_hook)
    rows = []
    try:
        for key, prompt in items:
            d = rec[key]
            toks = d["tokens"]
            st.update(step=-1, forced=toks, lp=[], t=[])
            sid = "df-" + hashlib.md5(key.encode()).hexdigest()[:8]
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
            w.active_session = sid
            torch.cuda.reset_peak_memory_stats()
            w.generate(prompt, max_new_tokens=len(toks), temperature=0.0, top_p=1.0,
                       repetition_penalty=1.0)
            peak = torch.cuda.max_memory_allocated() / 1e9
            n = min(len(st["lp"]), len(toks))
            kls, top1, dn = [], [], []
            for j in range(n):
                lpd = d["lp"][j].float().to(st["lp"][j].device)
                lpa = st["lp"][j]
                pd = lpd.exp()
                kls.append(float((pd * (lpd - lpa)).sum()))
                top1.append(int(int(lpa.argmax()) == toks[j]))
                dn.append(float(lpd[toks[j]] - lpa[toks[j]]))
            ts = st["t"]
            try:
                kv = dkv_kv_bytes(w.manager, d["ntok"], sid)
            except Exception as e:                               # noqa: BLE001
                kv = {"kv_error": str(e)}
            row = {"key": key, "ntok": d["ntok"], "steps": n,
                   "kl0": kls[0] if kls else None,
                   "kl": kls[1:], "top1": top1[1:], "dnll": dn[1:],
                   "peak_gb": peak,
                   "dec_tps": ((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 2 else None}
            row.update({k: kv.get(k) for k in ("kv_physical_gb", "kv_compression_x",
                                                "kv_error") if k in kv})
            row["resid_attn_used"] = getattr(w.manager, "_qres_used", 0)
            rows.append(row)
            print("  %s: kl %.4f  top1 %.2f  kv %s  peak %.2f" % (
                key, sum(row["kl"]) / max(1, len(row["kl"])),
                sum(row["top1"]) / max(1, len(row["top1"])),
                row.get("kv_physical_gb"), peak), flush=True)
            try:
                w.clear_session(sid)
            except Exception:                                    # noqa: BLE001
                pass
    finally:
        hh.remove()
        hp.remove()
    with open(out_path, "w") as f:
        json.dump(rows, f)


# ──────────────────────────────────────────────────────────── driver ──
def summarize(name, rows):
    import statistics as S
    kl = [x for r in rows for x in r["kl"]]
    t1 = [x for r in rows for x in r["top1"]]
    dn = [x for r in rows for x in r["dnll"]]
    kv = [r["kv_physical_gb"] for r in rows if r.get("kv_physical_gb")]
    cmpx = [r["kv_compression_x"] for r in rows if r.get("kv_compression_x")]
    tps = [r["dec_tps"] for r in rows if r.get("dec_tps")]
    kl0 = [r["kl0"] for r in rows if r.get("kl0") is not None]
    return {"arm": name, "steps": len(kl),
            "kl": S.mean(kl) if kl else None,
            "kl_median": S.median(kl) if kl else None,
            "top1": S.mean(t1) if t1 else None,
            "dnll": S.mean(dn) if dn else None,
            "kl0": S.mean(kl0) if kl0 else None,
            "kv_gb": S.mean(kv) if kv else None,
            "cmp": S.mean(cmpx) if cmpx else None,
            "peak_gb": max(r["peak_gb"] for r in rows) if rows else None,
            "dec_tps": S.mean(tps) if tps else None,
            "resid_attn_used": max((r.get("resid_attn_used") or 0) for r in rows) if rows else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--preset", default="mid")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--lb", nargs="*", default=LB_DEFAULT)
    ap.add_argument("--ruler", nargs="*", default=RULER_DEFAULT)
    ap.add_argument("--arms", nargs="+", default=["base"],
                    help="name or name:ENV=v,ENV2=v2")
    ap.add_argument("--out", default=os.path.join(REPO, "paper", "results", "diag",
                                                  "fidelity"))
    ap.add_argument("--role", default="")
    ap.add_argument("--dense", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.role == "dense":
        return run_dense(args, args.json)
    if args.role == "dkv":
        return run_dkv(args, args.dense, args.json)

    os.makedirs(args.out, exist_ok=True)
    tag = args.model.split("/")[-1]
    setkey = hashlib.md5(json.dumps([args.model, args.quant, args.gen, args.max_length,
                                     args.lb, args.ruler]).encode()).hexdigest()[:10]
    dense = os.path.join(args.out, "dense_%s_%s.pt" % (tag, setkey))
    common = ["--model", args.model, "--preset", args.preset, "--quant", args.quant,
              "--gen", str(args.gen), "--max-length", str(args.max_length),
              "--lb"] + args.lb + ["--ruler"] + args.ruler
    env0 = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if not os.path.exists(dense):
        print("[dense] building control ->", dense, flush=True)
        p = subprocess.run([sys.executable, __file__, "--role", "dense", "--json", dense]
                           + common, env=env0)
        if p.returncode or not os.path.exists(dense):
            raise SystemExit("dense control failed")
    summ = []
    for spec in args.arms:
        name, _, envs = spec.partition(":")
        env = dict(env0)
        for kv in filter(None, envs.split(",")):
            k, v = kv.split("=", 1)
            env[k] = v
        js = os.path.join(args.out, "%s_%s_%s_%s.json" % (tag, args.preset, name, setkey))
        print("[arm %s] %s" % (name, envs or "(shipped defaults)"), flush=True)
        t0 = time.time()
        if os.path.exists(js) and os.path.getmtime(js) > os.path.getmtime(dense):
            print("[arm %s] already done, reusing %s" % (name, js), flush=True)
            p = subprocess.CompletedProcess([], 0)
        else:
            p = subprocess.run([sys.executable, __file__, "--role", "dkv", "--dense", dense,
                                "--json", js] + common, env=env)
        if p.returncode or not os.path.exists(js):
            print("[arm %s] FAILED" % name, flush=True)
            continue
        with open(js) as f:
            s = summarize(name, json.load(f))
        s["env"], s["wall_s"] = envs, round(time.time() - t0)
        summ.append(s)
    print("\n%-12s %6s %8s %8s %6s %7s %6s %6s %7s %7s" % (
        "arm", "steps", "KL", "KLmed", "top1", "dNLL", "kvGB", "cmp", "peak", "dec/s"))
    for s in summ:
        f = lambda v, p: ("%" + p) % v if v is not None else "-"      # noqa: E731
        print("%-12s %6d %8s %8s %6s %7s %6s %6s %7s %7s" % (
            s["arm"], s["steps"], f(s["kl"], ".4f"), f(s["kl_median"], ".4f"),
            f(s["top1"], ".3f"), f(s["dnll"], "+.3f"), f(s["kv_gb"], ".3f"),
            f(s["cmp"], ".2f"), f(s["peak_gb"], ".2f"), f(s["dec_tps"], ".1f")))
        if s.get("resid_attn_used"):
            print("%-12s   resid-attn weighted blocks: %d" % ("", s["resid_attn_used"]))
    # Paired against the first arm, step by step (same prompt, same forced
    # prefix): per-step KL is heavy-tailed, so report the paired mean
    # difference with a bootstrap interval, plus how many steps diverge badly.
    if len(summ) > 1:
        import random
        ref_name = summ[0]["arm"]

        def steps(name):
            js = os.path.join(args.out, "%s_%s_%s_%s.json" % (tag, args.preset, name, setkey))
            with open(js) as fh:
                return {(r["key"], j): (r["kl"][j], r["top1"][j])
                        for r in json.load(fh) for j in range(len(r["kl"]))}
        ref = steps(ref_name)
        print("\npaired vs %s (bootstrap 10k, 95%%)" % ref_name)
        print("%-12s %6s %22s %9s %9s %8s" % ("arm", "n", "dKL [CI]", "KL>0.5", "ref>0.5",
                                              "dtop1"))
        for s in summ[1:]:
            cur = steps(s["arm"])
            ks = sorted(set(ref) & set(cur))
            d = [cur[k][0] - ref[k][0] for k in ks]
            rng = random.Random(1234)
            means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d)
                           for _ in range(10000))
            lo, hi = means[250], means[9749]
            big = sum(cur[k][0] > 0.5 for k in ks)
            bigr = sum(ref[k][0] > 0.5 for k in ks)
            dt = sum(cur[k][1] - ref[k][1] for k in ks) / len(ks)
            s["paired"] = {"n": len(ks), "dkl": sum(d) / len(d), "lo": lo, "hi": hi,
                           "big": big, "big_ref": bigr, "dtop1": dt}
            print("%-12s %6d %+8.4f [%+.4f,%+.4f] %9d %9d %+8.3f" % (
                s["arm"], len(ks), sum(d) / len(d), lo, hi, big, bigr, dt))
    with open(os.path.join(args.out, "summary_%s_%s.jsonl" % (tag, setkey)), "a") as f:
        for s in summ:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
