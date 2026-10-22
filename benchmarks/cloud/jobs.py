"""The Y1 cloud campaign: every GPU job, as data.

WHY A MANIFEST AND NOT .cmd FILES
---------------------------------
The V3-V6 campaigns were batch files run in order on one card. Spread over many
machines the same work needs a unit a machine can claim, finish and hand back:
a JOB, here one harness invocation with its own output files. Big runs are cut
into shards along an axis the harness's resume key already ignores (RULER by
task, LongBench by dataset), so shards of one store merge back exactly
(`ctl.py merge`).

WHAT IS MEASURED, AND WHY ON THESE CARDS
----------------------------------------
The paper's limitations section lists what one 12.9 GB consumer card could not
do. Each track answers one of them:

    q131   Qwen3.5-4B RULER at 131k, 20 items per task (was 5), all 13 tasks with
           RULER's answer prefix, and DENSE as a reference: on 80 GB the exact
           cache fits where it never did on the 12.9 GB card.
    g32 / g131  granite-4.2-8b RULER at 32k and 131k with dense beside DKV.
    sys    TTFT, decode rate, peak allocated/reserved and KV bytes vs context on
           a datacenter GPU, Linux, Inductor compiled (the 12.9 GB numbers were
           Windows/WDDM, partly eager).
    lad    the memory ladder on the same card, every arm, to each model's max.
    conc   sessions resident at once per arm: the serving question.
    lb     LongBench at the full official sample counts (was 20 per task).
    q262   RULER at Qwen3.5-4B's maximum context (262,144), never scored.
    q9b    a third model (Qwen3.5-9B), gated by the fidelity preflight.
    qshort Qwen3.5-4B 4k-64k with the answer prefix (the paper's short-length
           table, on the fixed harness).

DKV ON A BIG CARD IS NOT THE SAME CONFIGURATION UNLESS IT IS TOLD TO BE
------------------------------------------------------------------------
DKV_STREAM_ELASTIC keeps history exact while allocated memory plus
DKV_STREAM_ELASTIC_RESERVE_GB (default 3.5) stays under 94% of the card
(native_core/streaming_sparse_ingest._elastic_room). On an 80 GB card that
holds nearly everything exact, so "DKV" there would measure something close to
dense. The `card12` profiles set the reserve so the line falls exactly where it
did on the 12.878 GB card -- allocated >= 0.94*12.878 - 3.5 = 8.605 GB -- which
reproduces the paper's method with no code change (the decode fingerprint is
untouched). The `native` profiles leave the card's own line: what DKV does when
it has the memory, reported as a separate systems series.

Every job runs `smoke.<sig>` first: the same harness, arm, environment and
parameters at a tiny size. A configuration that cannot run fails in minutes on
the first machine that tries it, and its long jobs are never claimed.

    python benchmarks/cloud/jobs.py summary          # tracks, jobs, GPU-hours
    python benchmarks/cloud/jobs.py list [--track q131] [--smoke]
    python benchmarks/cloud/jobs.py show q131.dkv_stream.niah_single_1
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

CAMPAIGN = "Y1"
# Where every job writes; ctl.py merge turns these into paper/results/<sub>/Y1_*.
OUT_ROOT = "paper/results/cloud/" + CAMPAIGN

QW = "Qwen/Qwen3.5-4B"
GR = "ibm-granite/granite-4.2-8b"
Q9 = "Qwen/Qwen3.5-9B"
TAG = {QW: "Qwen3.5-4B", GR: "granite-4.2-8b", Q9: "Qwen3.5-9B"}

RULER_13 = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
            "niah_multikey_2", "niah_multikey_3", "niah_multivalue",
            "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]
LB_6 = ["qasper", "narrativeqa", "hotpotqa", "multifieldqa_en", "gov_report",
        "trec"]
# Items RULER data is generated with, whatever a job reads: every machine must
# hold the same items, and the generator's draws depend on the count.
RULER_NSAMP = 20
# Length labels as the paper writes them (131072 is "131k", not 128k).
LBL = {4096: "4k", 8192: "8k", 16384: "16k", 24576: "24k", 32768: "32k", 65536: "64k",
       131072: "131k", 262144: "262k"}

# ---------------------------------------------------------------- profiles --
# Shared by every job. RULER_ANSWER_PREFIX=1 is RULER as published (the paper's
# pre-10-20 stores lacked it); every arm of this campaign runs with it.
BASE_ENV = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1",
            "DKV_TRITON_STRICT": "1", "RULER_ANSWER_PREFIX": "1"}

# The paper's streaming DKV (V3_BEYOND / V6): low-rank store, SDPA prefill,
# elastic window, exact up to the model's auto threshold, streaming past it.
AUTO_TOKENS = {QW: "98304", GR: "16384", Q9: "98304"}
CARD12 = {"DKV_STREAM_ELASTIC_RESERVE_GB": "{card12_reserve_gb}"}


def dkv_env(profile: str, model: str) -> Dict[str, str]:
    stream = {"DKV_STORE": "lowrank", "DKV_PREFILL_SDPA": "1",
              "DKV_STREAM_ELASTIC": "1", "DKV_STREAMING_COMPRESS": "auto",
              "DKV_STREAM_AUTO_TOKENS": AUTO_TOKENS[model]}
    p = {
        "stream":    {**stream, **CARD12},
        "native":    dict(stream),
        "noquery":   {**stream, **CARD12, "DKV_RESIDUAL_NO_QUERY": "1"},
        "hybrid":    {**stream, **CARD12, "DKV_STORE": "hybrid"},
        "codes4":    {**stream, **CARD12, "DKV_STORE": "codes", "DKV_QUANT_BITS": "4"},
        "attendall": {**stream, **CARD12, "DKV_TOPK_BLOCKS": "0"},
        # Exact (lean) prefill, store built after it: the paper's dkv_exact.
        "exact":     {"DKV_STORE": "lowrank", "DKV_PREFILL_SDPA": "1",
                      "DKV_STREAM_ELASTIC": "1", "DKV_STREAMING_COMPRESS": "0",
                      "DKV_EXACT_LEAN": "1"},
        # LongBench's streaming row (V3): stream every prompt.
        "forced":    {"DKV_STORE": "lowrank", "DKV_PREFILL_SDPA": "1",
                      "DKV_STREAM_ELASTIC": "1", "DKV_STREAMING_COMPRESS": "1", **CARD12},
        # LongBench's dkv_mid row: the library's default path.
        "default":   {},
        # ---- the rest reproduce paper stores exactly as their .meta.json
        # recorded them (pre-V3 recipes carried no store/SDPA/elastic switches)
        # dkv_stream_fixed (BEYOND_*, TILED_*): auto streaming, no elastic window
        "fixed":     {"DKV_STREAMING_COMPRESS": "auto",
                      "DKV_STREAM_AUTO_TOKENS": AUTO_TOKENS[model]},
        # dkv_fstream_fixed (Qwen3.5-4B_dkv_mid_forcedstream_tiled_32k)
        "forced_fixed": {"DKV_STREAMING_COMPRESS": "1"},
        # dkv_allblocks (V4S_..._attendall_32k): exact, routing off
        "exact_attendall": {"DKV_STREAMING_COMPRESS": "0", "DKV_TOPK_BLOCKS": "0"},
        # dkv_codes4_all (V5_..._codes4_attendall_131k)
        "codes4_all": {**stream, **CARD12, "DKV_STORE": "codes", "DKV_QUANT_BITS": "4",
                       "DKV_TOPK_BLOCKS": "0"},
        # *_streamlow (LongBench dkv_stream_fixed, ladder dkv_stream_untiled).
        # Tiling was off in the code those ran on; it is on now, so this is the
        # recipe, not the old code state.
        "streamlow": {"DKV_PREFILL_LOWMEM": "1", "DKV_REMAT_CACHE": "0",
                      "DKV_STREAMING_COMPRESS": "1"},
        # LongBench dkv_qk / dkv_qk_stream (V3_..._hybrid_mid[_stream]_len12000)
        "hybrid_exact": {"DKV_PREFILL_SDPA": "1", "DKV_STORE": "hybrid",
                         "DKV_STREAM_ELASTIC": "1", **CARD12},
        "hybrid_forced": {"DKV_PREFILL_SDPA": "1", "DKV_STORE": "hybrid",
                          "DKV_STREAM_ELASTIC": "1", "DKV_STREAMING_COMPRESS": "1", **CARD12},
    }
    return p[profile]


# arm name in this campaign -> (harness --arm, baseline params, dkv profile)
SL = {"n_sink": 4, "recency_window": 2044}
H2 = {"budget": 2048, "recency_window": 512, "prefill_chunk": 256}
ARMS = {
    "dense":       ("dense", {"defrag": True}, None),
    "dense_tiled": ("dense_tiled", {}, None),
    "dense_static": ("dense", {"static": True}, None),
    "snapkv":      ("snapkv", {"budget": 2016, "window": 32, "defrag": True}, None),
    "kivi4_tiled": ("kivi4_tiled", {}, None),
    "kivi2_tiled": ("kivi2_tiled", {}, None),
    "sl_chunk":    ("streamingllm_chunked", SL, None),
    "h2o_chunk":   ("h2o_chunked", H2, None),
    # Eviction at 1.5x DKV/mid's bytes at 131k on Qwen3.5-4B (0.78 GB, V4S).
    "sl_eq":       ("streamingllm_chunked", {"n_sink": 4, "recency_window": 23804}, None),
    "snapkv_q_eq": ("snapkv_chunked_q", {"budget": 23808, "recency_window": 2048,
                                         "question_tail": 64}, None),
    "dkv_stream":  ("dkv", {}, "stream"),
    "dkv_native":  ("dkv", {}, "native"),
    "dkv_noquery": ("dkv", {}, "noquery"),
    "dkv_hybrid":  ("dkv", {}, "hybrid"),
    "dkv_codes4":  ("dkv", {}, "codes4"),
    "dkv_attendall": ("dkv", {}, "attendall"),
    "dkv_exact":   ("dkv", {}, "exact"),
    "dkv_forced":  ("dkv", {}, "forced"),
    "dkv_mid":     ("dkv", {}, "default"),
    # LongBench post-hoc arms, parameters as the paper's stores recorded them.
    "kivi4":       ("kivi4", {}, None),
    "kivi2":       ("kivi2", {}, None),
    "int8_kv":     ("int8_kv", {}, None),
    "snapkv_lb":   ("snapkv", {"budget": 2016, "window": 32}, None),
    "h2o_posthoc": ("h2o", {"budget": 2048, "recency_window": 512, "window": 32}, None),
    "sl_posthoc":  ("streamingllm", SL, None),
    # ---- arms that reproduce the remaining paper series
    "dense_plain": ("dense", {}, None),                  # ladder dense_shipped
    "dense_offload": ("dense", {"offload": True}, None),
    "kivi4_chunk": ("kivi4_chunked", {}, None),
    "dkv_high":    ("dkv", {}, "default"),               # preset high (see PRESET)
    "dkv_fixed":   ("dkv", {}, "fixed"),
    "dkv_forced_fixed": ("dkv", {}, "forced_fixed"),
    "dkv_attendall_exact": ("dkv", {}, "exact_attendall"),
    "dkv_codes4_all": ("dkv", {}, "codes4_all"),
    "dkv_streamlow": ("dkv", {}, "streamlow"),
    "dkv_hybrid_exact": ("dkv", {}, "hybrid_exact"),
    "dkv_hybrid_forced": ("dkv", {}, "hybrid_forced"),
}
PRESET = {"dkv_high": "high"}                            # every other DKV arm: mid

# ------------------------------------------------------------- cost model --
# Seconds per RULER item at 131k on Qwen3.5-4B, measured on the 12.9 GB card,
# divided by an assumed datacenter speedup. Only orders jobs and sizes ETAs;
# `ctl.py plan` rescales it from finished jobs.
SPEEDUP = 2.5
S131 = {"dkv": 270, "dense": 140, "kivi4_tiled": 260, "kivi2_tiled": 260,
        "streamingllm_chunked": 200, "h2o_chunked": 420, "snapkv": 160,
        "snapkv_chunked_q": 240, "dense_tiled": 200}
MODEL_X = {QW: 1.0, GR: 2.0, Q9: 1.9}


def ruler_item_s(model: str, harm: str, length: int) -> float:
    return (S131.get(harm, 270) * MODEL_X[model] * (length / 131072) ** 1.1 + 4) / SPEEDUP


# ------------------------------------------------------------------ jobs --
class Job(dict):
    """A dict so it serializes as-is. Keys:
    id, track, prio (lower runs first), argv (repo-relative, run with the
    worker's python), env, outputs (files the harness writes; pushed back),
    est_h, min_vram_gb, needs (job ids), data ([model, length] RULER sets),
    timing (bool: numbers are latencies, run alone on the GPU), sig.
    """


def _sig(harness: str, model: str, arm: str, env: Dict[str, str], params) -> str:
    blob = json.dumps([harness, model, arm, env, params], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _job(track, prio, jid, harness, model, arm, argv, outs, est_h, vram,
         data=None, timing=False, env=None, raw=False):
    """raw: argv is the harness's whole command line (no --model and
    --baseline-params added), env is given. An output ending in "/" is a
    directory: its *.json / *.jsonl files are pushed."""
    harm, params, prof = ARMS.get(arm, (arm, {}, None))
    if env is None:
        env = {**BASE_ENV, **(dkv_env(prof, model) if prof else {})}
    full = ["benchmarks/" + harness] + ([] if raw else ["--model", model]) + argv
    if not raw and harness != "bench_concurrency_cuda.py":
        full += ["--baseline-params", json.dumps(params)]
    sig = _sig(harness, model, arm, env, params if not raw else argv)
    return Job(id=jid, track=track, prio=prio, argv=full, env=env,
               outputs=outs + [o + ".meta.json" for o in outs if not o.endswith("/")],
               est_h=round(est_h, 2),
               min_vram_gb=vram, needs=["smoke." + sig], data=data or [],
               timing=timing, sig=sig, model=model, arm=arm, harness=harness)


def _arm_flags(arm: str, multi: bool = False) -> List[str]:
    harm = ARMS[arm][0]
    flags = ["--arms" if multi else "--arm", harm]
    if harm == "dkv":
        flags += ["--preset", PRESET.get(arm, "mid")]
    return flags


def ruler_jobs(track, prio, model, length, arms, per_task, vram):
    jobs = []
    for arm in arms:
        harm = ARMS[arm][0]
        stem = f"{TAG[model]}_{arm}_{LBL[length]}"
        for t in RULER_13:
            out = f"{OUT_ROOT}/ruler/{stem}/{t}.jsonl"
            argv = _arm_flags(arm) + [
                "--data", "{ruler_data}/" + TAG[model],
                "--min-length", str(length), "--max-length", str(length),
                "--per-task", str(per_task), "--tasks", t, "--out", out]
            est = per_task * ruler_item_s(model, harm, length) / 3600
            jobs.append(_job(track, prio, f"{track}.{arm}.{t}", "run_ruler_cuda.py",
                             model, arm, argv, [out], est, vram,
                             data=[[model, length]]))
    return jobs


def ruler_short_jobs(track, prio, model, lengths, arms, vram):
    """One job per (arm, length), every task: short items are cheap."""
    jobs = []
    for arm in arms:
        harm = ARMS[arm][0]
        for L in lengths:
            stem = f"{TAG[model]}_{arm}_{LBL[L]}"
            out = f"{OUT_ROOT}/ruler/{stem}/all.jsonl"
            argv = _arm_flags(arm) + [
                "--data", "{ruler_data}/" + TAG[model],
                "--min-length", str(L), "--max-length", str(L), "--out", out]
            est = 13 * RULER_NSAMP * ruler_item_s(model, harm, L) / 3600
            jobs.append(_job(track, prio, f"{track}.{arm}.{LBL[L]}",
                             "run_ruler_cuda.py", model, arm, argv, [out], est, vram,
                             data=[[model, L]]))
    return jobs


def longbench_jobs(track, prio, model, arms, n, vram):
    jobs = []
    for arm in arms:
        for ds in LB_6:
            out = f"{OUT_ROOT}/longbench/{TAG[model]}_{arm}_len12000/{ds}.jsonl"
            argv = _arm_flags(arm) + ["--max-length", "12000", "--num-samples", str(n),
                                      "--datasets", ds, "--out", out]
            est = n * (14 if ARMS[arm][0] != "h2o_chunked" else 30) * MODEL_X[model] / SPEEDUP / 3600
            jobs.append(_job(track, prio, f"{track}.{arm}.{ds}", "run_longbench_cuda.py",
                             model, arm, argv, [out], est, vram))
    return jobs


def systems_jobs(track, prio, model, plan, vram):
    jobs = []
    for arm, ctxs, est in plan:
        out = f"{OUT_ROOT}/systems/{TAG[model]}_{arm}.jsonl"
        argv = _arm_flags(arm, multi=True) + [
            "--gen", "64", "--reps", "3", "--timeout", "7200",
            "--contexts"] + [str(c) for c in ctxs] + ["--out", out]
        jobs.append(_job(track, prio, f"{track}.{TAG[model]}.{arm}", "bench_systems_cuda.py",
                         model, arm, argv, [out], est, vram, timing=True))
    return jobs


def ladder_jobs(track, prio, model, plan, vram):
    jobs = []
    for arm, ctxs, est in plan:
        out = f"{OUT_ROOT}/ladder/{TAG[model]}_{arm}.jsonl"
        argv = _arm_flags(arm, multi=True) + [
            "--gen", "128", "--timeout", "7200",
            "--contexts"] + [str(c) for c in ctxs] + ["--out", out]
        jobs.append(_job(track, prio, f"{track}.{TAG[model]}.{arm}", "context_ladder.py",
                         model, arm, argv, [out], est, vram))
    return jobs


def concurrency_jobs(track, prio, model, ctx, arms, n_max, est, vram):
    jobs = []
    for arm in arms:
        out = f"{OUT_ROOT}/concurrency/{TAG[model]}_ctx{ctx}_{arm}.jsonl"
        argv = ["--arms", ARMS[arm][0], "--ctx", str(ctx), "--n-max", str(n_max),
                "--gen", "8", "--dkv-pool-gb", "{pool_gb}", "--timeout", "21600",
                "--out", out]
        jobs.append(_job(track, prio, f"{track}.{TAG[model]}.{LBL[ctx]}.{arm}",
                         "bench_concurrency_cuda.py", model, arm, argv, [out], est, vram))
    return jobs


def snapshot_jobs(track, prio):
    """The paper's allocator traces (paperdata.trace): the named sites alive at
    peak, at two lengths, so growth per 1k tokens is per code site."""
    variants = [
        ("base", {}, [(GR, "dkv", 8192), (GR, "dkv", 16384), (GR, "dense", 8192),
                      (GR, "dense", 16384), (QW, "dkv", 16384), (QW, "dkv", 32768)]),
        ("noremat", {"DKV_REMAT_CACHE": "0"}, [(GR, "dkv", 8192), (GR, "dkv", 16384)]),
        ("streaming", {"DKV_REMAT_CACHE": "0", "DKV_STREAMING_COMPRESS": "1"},
         [(GR, "dkv", 6144), (GR, "dkv", 10240)]),
        ("streaming_lowmem", {"DKV_PREFILL_LOWMEM": "1", "DKV_REMAT_CACHE": "0",
                              "DKV_STREAMING_COMPRESS": "1"},
         [(GR, "dkv", 6144), (GR, "dkv", 10240)]),
    ]
    jobs = []
    for name, env, pts in variants:
        out = f"{OUT_ROOT}/diag/granite_peak_snapshot_{name}.jsonl"
        argv = ["--out", out, "--points"] + [f"{m}|{a}|{c}" for m, a, c in pts]
        jobs.append(_job(track, prio, f"{track}.snapshot.{name}", "diag_granite_snapshot.py",
                         GR, "snapshot_" + name, argv, [out], 0.5, 24,
                         env={**BASE_ENV, **env}, raw=True))
    return jobs


def fidelity_jobs(track, prio):
    """Decode fidelity (dense's greedy answer forced through each arm, KL per
    step) for the checks the paper quotes: routing 16 records against every
    record with the codes store on granite at 24k-32k, and the exact, forced
    streaming and hybrid paths on Qwen3.5-4B. One job per model."""
    hard = ["niah_multikey_2", "niah_multikey_3", "qa_1", "qa_2"]
    g_stream = ("DKV_PREFILL_SDPA=1,DKV_STREAM_ELASTIC=1,DKV_STREAMING_COMPRESS=auto,"
                "DKV_STREAM_AUTO_TOKENS=16384,DKV_STREAM_ELASTIC_RESERVE_GB={card12_reserve_gb}")
    plan = [
        (GR, [24576, 32768], {
            "codes": "DKV_STORE=codes,DKV_QUANT_BITS=4," + g_stream,
            "codes_all": "DKV_STORE=codes,DKV_QUANT_BITS=4,DKV_TOPK_BLOCKS=0," + g_stream,
            "stream": "DKV_STORE=lowrank," + g_stream}),
        (QW, [32768], {
            "exact": "DKV_STORE=lowrank,DKV_PREFILL_SDPA=1,DKV_STREAM_ELASTIC=1,"
                     "DKV_STREAMING_COMPRESS=0,DKV_EXACT_LEAN=1",
            "forced": "DKV_STORE=lowrank,DKV_PREFILL_SDPA=1,DKV_STREAM_ELASTIC=1,"
                      "DKV_STREAMING_COMPRESS=1,DKV_STREAM_ELASTIC_RESERVE_GB={card12_reserve_gb}",
            "hybrid": "DKV_STORE=hybrid,DKV_PREFILL_SDPA=1,DKV_STREAM_ELASTIC=1,"
                      "DKV_STREAMING_COMPRESS=0"}),
    ]
    jobs = []
    for model, lengths, arms in plan:
        out = f"{OUT_ROOT}/diag/fidelity_{TAG[model]}/"
        argv = ["--model", model, "--gen", "24", "--out", out, "--lb",
                "qasper:15-15", "multifieldqa_en:3-3", "hotpotqa:1-1", "--ruler"]
        argv += [f"{L}/{t}:0-4" for L in lengths for t in hard]
        argv += ["--arms"] + [f"{k}:{v}" for k, v in arms.items()]
        jobs.append(_job(track, prio, f"{track}.decode.{TAG[model]}", "decode_fidelity.py",
                         model, "fidelity", argv, [out], 3.0, 40,
                         env={**BASE_ENV, "DKV_RULER_DATA": "{ruler_data}"}, raw=True,
                         data=[[model, L] for L in lengths]))
    return jobs


Q_LADDER = [8192, 32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144]
G_LADDER = [4096, 16384, 32768, 49152, 65536, 98304, 131072]
LADDER_ARMS = ("dense", "dense_plain", "dense_tiled", "dense_static", "dense_offload",
               "snapkv", "kivi4_tiled", "kivi2_tiled", "kivi4_chunk", "sl_chunk", "h2o_chunk",
               "dkv_mid", "dkv_exact", "dkv_stream", "dkv_native", "dkv_fixed",
               "dkv_streamlow", "dkv_hybrid", "dkv_codes4")
SYSTEMS_ARMS = (("dense", 2.0), ("dense_static", 2.0), ("dense_tiled", 2.0),
                ("dense_offload", 3.0), ("snapkv", 2.0), ("kivi4_tiled", 2.5),
                ("kivi2_tiled", 2.5), ("sl_chunk", 2.0), ("h2o_chunk", 4.0),
                ("dkv_exact", 2.5), ("dkv_stream", 3.0), ("dkv_native", 3.0),
                ("dkv_codes4", 3.0), ("dkv_hybrid", 3.0))


def build() -> List[Job]:
    """Every series the paper reads (paperdata.py: LADDERS, RULER, LONGBENCH,
    SYSTEMS, trace; the decode-fidelity checks) gets a cloud twin, plus what
    the 12.9 GB card could not run. P1 the headline numbers, P2 every other
    paper series, P3 extensions."""
    J: List[Job] = []
    # ---- P1: headline quality past the exact limit, at sizes that resolve
    J += ruler_jobs("q131", 1, QW, 131072,
                    ["dense", "dkv_stream", "kivi4_tiled", "sl_chunk", "h2o_chunk",
                     "sl_eq", "snapkv_q_eq", "dkv_noquery", "dkv_codes4", "kivi2_tiled",
                     "dkv_hybrid", "snapkv", "dkv_native", "dkv_fixed", "dkv_codes4_all",
                     "dkv_attendall"], 20, 40)
    # ---- P1: systems on a datacenter card (MLSys's own question)
    qs = [8192, 32768, 65536, 131072, 196608, 262144]
    gs = [4096, 16384, 32768, 65536, 131072]
    J += systems_jobs("sys", 1, QW, [(a, qs, e) for a, e in SYSTEMS_ARMS], 70)
    J += systems_jobs("sys", 1, GR, [(a, gs, e) for a, e in SYSTEMS_ARMS], 70)
    for m in (QW, GR):
        for c in (16384, 32768, 131072):
            J += concurrency_jobs("conc", 1, m, c,
                                  ["dense", "dkv_mid", "snapkv", "sl_chunk", "h2o_chunk"],
                                  256 if c <= 32768 else 128,
                                  1.5 if c <= 32768 else 4.0, 70)
    # ---- P2: every other RULER series of the paper, on the fixed harness.
    # Qwen3.5-4B 4k-64k: the main table (dense, dkv_high, dkv_mid, SnapKV,
    # chunk-wise eviction, hybrid, tiled KIVI) plus streaming DKV.
    J += ruler_short_jobs("qshort", 2, QW, [4096, 8192, 16384, 32768, 65536],
                          ["dense", "dkv_mid", "dkv_high", "dkv_stream", "snapkv",
                           "sl_chunk", "h2o_chunk", "dkv_hybrid", "kivi4_tiled",
                           "kivi2_tiled"], 24)
    # Qwen3.5-4B 32k ablations: forced streaming (elastic and fixed window),
    # routing off on the exact path.
    J += ruler_short_jobs("q32x", 2, QW, [32768],
                          ["dkv_forced", "dkv_forced_fixed", "dkv_attendall_exact"], 24)
    # granite: dense to 16k (the paper's dense row), past the 12.9 GB ceiling
    # at 24k and 32k (the paper's beyond-limit pair), and to its 131k maximum
    # (the paper ran 114,688 there because that was what fit).
    J += ruler_short_jobs("gshort", 2, GR, [4096, 8192, 16384],
                          ["dense", "dkv_mid", "dkv_stream", "kivi4_tiled", "snapkv"], 24)
    for L, tr in ((24576, "g24"), (32768, "g32")):
        J += ruler_jobs(tr, 2, GR, L,
                        ["dense", "dkv_stream", "dkv_fixed", "kivi4_tiled", "kivi2_tiled",
                         "sl_chunk", "h2o_chunk", "dkv_codes4", "dkv_hybrid", "snapkv"], 20, 40)
    J += ruler_jobs("g131", 2, GR, 131072,
                    ["dense", "dkv_stream", "kivi4_tiled", "sl_chunk", "h2o_chunk",
                     "dkv_codes4", "kivi2_tiled", "dkv_hybrid"], 10, 70)
    # ---- P2: LongBench, every arm of the paper's table, 200 items/dataset
    J += longbench_jobs("lb", 2, GR,
                        ["dense", "dkv_mid", "dkv_high", "dkv_forced", "dkv_streamlow",
                         "dkv_hybrid_exact", "dkv_hybrid_forced", "kivi4_tiled",
                         "kivi2_tiled", "snapkv_lb", "sl_chunk", "h2o_chunk", "sl_posthoc",
                         "h2o_posthoc", "kivi4", "kivi2", "int8_kv", "kivi4_chunk"], 200, 24)
    # ---- P2: memory ladders (every ladder series), traces, decode fidelity
    J += ladder_jobs("lad", 2, QW, [(a, Q_LADDER, 2.5) for a in LADDER_ARMS], 70)
    J += ladder_jobs("lad", 2, GR, [(a, G_LADDER, 2.5) for a in LADDER_ARMS], 70)
    J += snapshot_jobs("diag", 2)
    J += fidelity_jobs("diag", 2)
    # ---- P3: max context, a third model
    J += ruler_jobs("q262", 3, QW, 262144,
                    ["dense", "dkv_stream", "kivi4_tiled", "sl_chunk", "dkv_codes4"], 10, 70)
    J += ruler_jobs("q9b", 3, Q9, 131072,
                    ["dense", "dkv_stream", "kivi4_tiled", "sl_chunk", "h2o_chunk"], 10, 40)
    J += systems_jobs("q9b", 3, Q9, [("dense", qs, 2.5), ("dkv_stream", qs, 3.5),
                                     ("kivi4_tiled", qs, 3.0)], 70)
    return J + smoke_jobs(J)


# ----------------------------------------------------------------- smoke --
def _smoke_argv(j: Job) -> List[str]:
    """The job's own argv at a size that runs in minutes on any card."""
    a = list(j["argv"])

    def setv(flag, vals):
        i = a.index(flag)
        k = i + 1
        while k < len(a) and not a[k].startswith("--"):
            k += 1
        a[i + 1:k] = vals

    def drop(flag):
        if flag in a:
            i = a.index(flag)
            k = i + 1
            while k < len(a) and not a[k].startswith("--"):
                k += 1
            del a[i:k]

    out = f"{OUT_ROOT}/smoke/{j['sig']}.jsonl"
    h = j["harness"]
    if h == "run_ruler_cuda.py":
        setv("--min-length", ["8192"]); setv("--max-length", ["8192"])
        drop("--per-task"); a += ["--per-task", "1"]
        drop("--tasks"); a += ["--tasks", "niah_single_1", "qa_1"]
    elif h == "run_longbench_cuda.py":
        setv("--num-samples", ["1"]); setv("--max-length", ["4000"])
        setv("--datasets", ["qasper"])
    elif h == "bench_systems_cuda.py":
        setv("--contexts", ["8192"]); setv("--reps", ["1"]); setv("--gen", ["8"])
    elif h == "context_ladder.py":
        setv("--contexts", ["4096", "8192"]); setv("--gen", ["8"])
    elif h == "bench_concurrency_cuda.py":
        setv("--ctx", ["4096"]); setv("--n-max", ["2"])
    elif h == "diag_granite_snapshot.py":
        i = a.index("--points") + 1
        # every (model, arm) of the job once, plus a second length for a slope
        pairs = list(dict.fromkeys(p.rsplit("|", 1)[0] for p in a[i:]))
        a[i:] = [p + "|4096" for p in pairs] + [pairs[0] + "|6144"]
    elif h == "decode_fidelity.py":
        setv("--gen", ["4"]); setv("--lb", ["qasper:15-15"])
        setv("--ruler", ["8192/qa_1:0-0"])
        out = out[:-len(".jsonl")] + "/"
    setv("--out", [out])
    return a


def smoke_jobs(jobs: List[Job]) -> List[Job]:
    seen: Dict[str, Job] = {}
    for j in jobs:
        if j["sig"] in seen:
            continue
        fid = j["harness"] == "decode_fidelity.py"
        out = f"{OUT_ROOT}/smoke/{j['sig']}" + ("/" if fid else ".jsonl")
        data = ([[j["model"], 8192]] if j["harness"] in ("run_ruler_cuda.py",
                                                       "decode_fidelity.py") else [])
        env = dict(j["env"])
        # Streaming starts at the auto threshold (98k / 16k); at smoke sizes it
        # would never run. 4096 puts an 8k prompt through the streaming path.
        if env.get("DKV_STREAMING_COMPRESS") == "auto":
            env["DKV_STREAM_AUTO_TOKENS"] = "4096"
        seen[j["sig"]] = Job(
            id="smoke." + j["sig"], track="smoke", prio=0, argv=_smoke_argv(j),
            env=env, outputs=[out] if fid else [out, out + ".meta.json"], est_h=0.1,
            min_vram_gb=12, needs=[], data=data, timing=False, sig=j["sig"],
            model=j["model"], arm=j["arm"], harness=j["harness"],
            smoke_for=j["id"])
    return list(seen.values())


def check(jobs: List[Job]) -> None:
    """Ids and output files must be unique: two jobs writing one file would
    interleave rows from two configurations, which is what checkpoint.py's
    config guard exists to stop and cannot see across machines."""
    ids, outs = set(), {}
    for j in jobs:
        assert j["id"] not in ids, "duplicate job id " + j["id"]
        ids.add(j["id"])
        for o in j["outputs"]:
            assert o not in outs, f"{o} written by {outs[o]} and {j['id']}"
            outs[o] = j["id"]
    for j in jobs:
        for n in j["needs"]:
            assert n in ids, f"{j['id']} needs unknown {n}"


def all_jobs() -> List[Job]:
    J = build()
    check(J)
    return J


def by_id() -> Dict[str, Job]:
    return {j["id"]: j for j in all_jobs()}


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "list", "show"])
    ap.add_argument("id", nargs="?")
    ap.add_argument("--track", default="")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    J = all_jobs()
    if a.cmd == "show":
        print(json.dumps(by_id()[a.id], indent=2))
        return
    if a.cmd == "list":
        for j in J:
            if (a.track and j["track"] != a.track) or (j["track"] == "smoke") != a.smoke:
                continue
            print(f"{j['id']:<48} p{j['prio']}  {j['est_h']:>5.2f} h  >= {j['min_vram_gb']} GB")
        return
    rows: Dict[str, List[float]] = {}
    for j in J:
        rows.setdefault(j["track"], []).append(j["est_h"])
    print(f"{'track':<8} {'jobs':>5} {'GPU-h':>7}")
    for t, v in sorted(rows.items(), key=lambda kv: min(
            j["prio"] for j in J if j["track"] == kv[0])):
        print(f"{t:<8} {len(v):>5} {sum(v):>7.1f}")
    print(f"{'total':<8} {len(J):>5} {sum(j['est_h'] for j in J):>7.1f}")


if __name__ == "__main__":
    main()
