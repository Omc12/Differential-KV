#!/usr/bin/env python3
"""LongBench on CUDA, run to the published protocol, with resumable checkpoints.

WHAT MAKES THIS THE REAL THING
------------------------------
Every piece of the evaluation protocol is the one the LongBench authors ship,
vendored under benchmarks/longbench_official/ and used unmodified:

  * data          the official `data.zip` from the LongBench dataset repo
                  (zai-org/LongBench, formerly THUDM/LongBench). The HF loading
                  SCRIPT cannot be used -- `datasets` 4.x refuses script-based
                  datasets -- so the archive is fetched and its per-task JSONL
                  read directly. Same bytes the script would have handed back.
  * prompts       config/dataset2prompt.json, verbatim
  * gen length    config/dataset2maxlen.json, verbatim
  * truncation    middle-truncation to --max-length, i.e. first half + last
                  half, because both ends carry instructions
  * chat template applied EXCEPT on trec/triviaqa/samsum/lsht/lcc/repobench-p,
                  which the authors exclude
  * scoring       metrics.py, imported and called, not reimplemented
  * post-proc     first line only on trec/triviaqa/samsum/lsht

Nothing here is tuned, and no preset knob is set by hand: the DKV arm takes a
shipped preset verbatim.

ARMS
----
  dense                    plain HF, no KV compression — the control
  dkv                      DKV at --preset
  streamingllm snapkv h2o  eviction baselines
  kivi2 kivi4 int8_kv      quantized-KV baselines
                           (see kv_baselines.py for what is faithful and what
                            is approximate — the memory axis differs by method)

CHECKPOINTING
-------------
Results append to JSONL and fsync per item; a rerun skips what is already
there. A power cut costs one sample. The run config is pinned in a sidecar
.meta.json and a mismatch on resume is fatal, so rows measured under different
settings can never be merged into one table.

USAGE
    python benchmarks/run_longbench_cuda.py --model ibm-granite/granite-4.2-8b \
        --arm dkv --preset mid --quant nf4 --max-length 15500 \
        --datasets qasper narrativeqa hotpotqa multifieldqa_en gov_report passage_retrieval_en \
        --num-samples 50 --out paper/results/longbench/granite_dkv_mid.jsonl

    # then, to score whatever is on disk (any subset, any time):
    python benchmarks/run_longbench_cuda.py --score paper/results/longbench/*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
OFFICIAL = os.path.join(HERE, "longbench_official")

sys.path.insert(0, HERE)
from checkpoint import ResumableJSONL                            # noqa: E402
from code_fingerprint import decode_fingerprint          # noqa: E402

# The English subset. The Chinese tasks (dureader/vcsum/lsht/multifieldqa_zh/
# passage_retrieval_zh) are excluded deliberately: none of the models in this
# study is a Chinese-capable release, so their scores would measure the model's
# language coverage rather than the KV method.
DEFAULT_DATASETS = ["qasper", "narrativeqa", "hotpotqa", "multifieldqa_en",
                    "gov_report", "passage_retrieval_en"]

# The authors' own exclusion list: chat models score better WITHOUT a chat
# template on these, so pred.py skips it.
NO_CHAT_TEMPLATE = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

# Whether cl.exe was available, i.e. whether Inductor could actually compile the
# decode path. Recorded PER RECORD rather than in the run config on purpose:
# quality is identical either way ("correct but unfused"), so quality rows from
# fused and unfused runs may be pooled, but LATENCY rows may not. Putting it in
# the config would instead invalidate perfectly good quality data on resume.
MSVC_OK = False
# eval.py keeps only the first line for these.
FIRST_LINE_ONLY = {"trec", "triviaqa", "samsum", "lsht"}


# ─────────────────────────────────────────────────────────────────────────────
# Official assets
# ─────────────────────────────────────────────────────────────────────────────

def official_config():
    with open(os.path.join(OFFICIAL, "config", "dataset2prompt.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    with open(os.path.join(OFFICIAL, "config", "dataset2maxlen.json"), encoding="utf-8") as f:
        maxlen = json.load(f)
    return prompts, maxlen


def official_metrics():
    sys.path.insert(0, OFFICIAL)
    import metrics as M
    return {
        "narrativeqa": M.qa_f1_score, "qasper": M.qa_f1_score,
        "multifieldqa_en": M.qa_f1_score, "multifieldqa_zh": M.qa_f1_zh_score,
        "hotpotqa": M.qa_f1_score, "2wikimqa": M.qa_f1_score,
        "musique": M.qa_f1_score, "dureader": M.rouge_zh_score,
        "gov_report": M.rouge_score, "qmsum": M.rouge_score,
        "multi_news": M.rouge_score, "vcsum": M.rouge_zh_score,
        "trec": M.classification_score, "triviaqa": M.qa_f1_score,
        "samsum": M.rouge_score, "lsht": M.classification_score,
        "passage_retrieval_en": M.retrieval_score, "passage_count": M.count_score,
        "passage_retrieval_zh": M.retrieval_zh_score, "lcc": M.code_sim_score,
        "repobench-p": M.code_sim_score,
    }


def longbench_data_dir() -> str:
    """Official data.zip, downloaded once and extracted beside itself."""
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    out = os.path.join(os.path.dirname(zp), "extracted")
    marker = os.path.join(out, "data")
    if not os.path.isdir(marker):
        os.makedirs(out, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(out)
    return marker


def load_task(name: str, n: int) -> List[Dict[str, Any]]:
    path = os.path.join(longbench_data_dir(), f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"LongBench task file missing: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction — the published recipe
# ─────────────────────────────────────────────────────────────────────────────

def apply_template(tokenizer, prompt: str, thinking: bool):
    """Chat template, with reasoning-mode models pinned to direct answers.

    granite-4.2 and Qwen3.5 both open a `<think>` block by default, so a
    LongBench run against them scores whatever fits in the task's generation
    budget of chain-of-thought -- qasper allows 128 tokens, and the model is
    still reasoning when it runs out. Measured on granite before this was
    fixed: every prediction began "Okay, let's tackle this question:" and no
    answer was ever reached.

    That is not a fair reading of the model and it is not what LongBench's
    per-task generation lengths were calibrated against, so thinking is turned
    OFF by default and the flag is recorded in the run config. It is applied
    identically to every arm, so it cannot advantage one of them.
    """
    msgs = [{"role": "user", "content": prompt}]
    if not thinking:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except (TypeError, ValueError):
            # Template does not take the kwarg; fall through to the default.
            pass
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)


def build_prompt(tokenizer, row, dataset, prompt_fmt, max_length,
                 thinking: bool = False):
    prompt = prompt_fmt.format(**row)
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) > max_length:
        # Middle truncation: the authors keep both ends because instructions sit
        # at the start AND the end of these prompts. Cutting the tail instead
        # removes the question.
        half = int(max_length / 2)
        prompt = (tokenizer.decode(ids[:half], skip_special_tokens=True)
                  + tokenizer.decode(ids[-half:], skip_special_tokens=True))
    if dataset not in NO_CHAT_TEMPLATE:
        try:
            prompt = apply_template(tokenizer, prompt, thinking)
        except Exception:                                        # noqa: BLE001
            pass
    return prompt


def post_process(pred: str, dataset: str) -> str:
    if dataset in FIRST_LINE_ONLY:
        pred = pred.lstrip("\n").split("\n")[0]
    return pred


def strip_prompt_echo(text: str, prompt: str, tok, budget_tokens: int = 0):
    """DKVHFWrapper.generate() returns PROMPT + COMPLETION, not the completion.

    hf_dkv_wrapper.py builds its output as `generated = prompt_ids.copy()` and
    decodes the whole list, so the string that comes back opens with the entire
    context. Scored as-is against a LongBench gold answer, a 34,000-character
    prompt echo produces a token-F1 near zero for every sample, and the arm
    looks catastrophically broken when nothing is wrong with it. (The MLX
    LongBench harness carries the same guard for mlx_lm, for the same reason.)

    The prompt is re-decoded through skip_special_tokens so the prefix matches
    what the wrapper produced, chat-template markers and all. Returns
    (completion, matched) -- `matched` false means the prefix was NOT found and
    the caller should keep the flag in the record rather than silently scoring
    something that may still contain context.
    """
    if not text:
        return text, True
    try:
        clean = tok.decode(tok(prompt).input_ids, skip_special_tokens=True)
    except Exception:                                            # noqa: BLE001
        return text, False
    if clean and text.startswith(clean):
        return text[len(clean):].lstrip(), True
    # _normalize_references() can rewrite the decoded string, so an exact
    # prefix match is not guaranteed. Fall back to the tail after the longest
    # matching head, and record that this happened.
    lo, hi = 0, min(len(clean), len(text))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text.startswith(clean[:mid]):
            lo = mid
        else:
            hi = mid - 1
    if lo > 200:                     # a real, substantial prompt echo
        text, ok = text[lo:].lstrip(), False
    else:
        ok = True

    # HARD BOUND: the model was asked for at most `max_new_tokens`. Anything
    # longer than that is echoed context by definition, whatever the prefix
    # search concluded -- this is arithmetic, not a heuristic.
    #
    # Measured on granite dkv/high: 2 of 120 items came back as 45,000
    # characters of the story itself, because _normalize_references() had
    # rewritten the head enough to break the prefix match. Those two dragged
    # narrativeqa's mean generation length from ~124 tokens to 1,118 against a
    # cap of 128. They scored ~0 either way, so they did not change the
    # headline -- but an unbounded field called "text" is how a harness bug
    # becomes a quality finding, and the next one might not be so harmless.
    #
    # What the model generated is at the END, after the echo, so the tail is
    # what survives.
    if budget_tokens:
        try:
            ids = tok(text, add_special_tokens=False).input_ids
            if len(ids) > budget_tokens:
                text = tok.decode(ids[-budget_tokens:], skip_special_tokens=True)
                ok = False
        except Exception:                                        # noqa: BLE001
            pass
    return text, ok


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def dkv_kv_bytes(mgr, seq_len: int, sid: str) -> Dict[str, float]:
    """DKV's KV footprint, on the same axis as the baselines' kv_physical_gb.

    Ported from run_a100_paper_experiments.py::analytic_kv_bytes, which is the
    GPU-validated accounting: it reads the corrected `mgr.sessions` view and the
    real pool allocation, NOT the legacy `manager.session_blocks` dict, which is
    empty on CUDA because the streaming path stores blocks elsewhere.

    Geometry comes off the manager rather than being derived from the config --
    on the hybrid models here, deriving head_dim from hidden_size/num_heads is
    wrong (see the head-geometry resolvers in the wrapper).
    """
    L, Hkv, d = mgr.num_layers, mgr.kv_heads, mgr.head_dim
    fp16 = 2
    B = int(getattr(mgr, "micro_block_size", 0) or getattr(mgr, "block_size", 64))
    pool = getattr(mgr, "native_pool", None)
    r = int(getattr(pool, "rank", None) or mgr.rank)

    kv_tok = Hkv * d * fp16 * 2
    lowrank_block = (B * r * 1 + 2 * Hkv * r * d * fp16 + 2 * Hkv * d * fp16 + 8)

    s0 = mgr.sessions.get(sid) or {}
    nb = ((s0.get("num_blocks") or [0])[0]) or (seq_len // B) or 1
    cfg = getattr(mgr, "config", None)
    max_dense = int(getattr(cfg, "max_active_dense_tokens", None)
                    or getattr(mgr, "max_active_dense_tokens", 1024) or 1024)
    raw_dl = (s0.get("dense_lens") or [0])[0]
    dl = min(raw_dl, max_dense) if raw_dl > 0 else min(seq_len, max_dense)
    res_n0 = (s0.get("comp_res_n") or [[]])[0][:nb]
    preset = str(getattr(cfg, "preset", None) or os.environ.get("DKV_PRESET", "mid")).lower()
    res_cap = int(getattr(cfg, "max_residual_tokens", None)
                  or {"low": 40, "mid": 64, "high": 128}.get(preset, 64))
    res_used = sum(min(int(x), res_cap) for x in res_n0) if res_n0 else nb * res_cap

    store_used = L * (nb * lowrank_block + res_used * kv_tok + dl * kv_tok)
    pool_physical = 0
    if pool is not None and hasattr(pool, "_pool_mb"):
        try:
            pool_physical = int(pool._pool_mb() * 1024 ** 2)
        except Exception:                                        # noqa: BLE001
            pool_physical = 0
    dense_equiv = L * seq_len * kv_tok
    return {
        "kv_physical_gb": store_used / 1e9,
        "kv_pool_physical_gb": pool_physical / 1e9,
        "kv_dense_equiv_gb": dense_equiv / 1e9,
        "kv_compression_x": (dense_equiv / store_used) if store_used else 0.0,
        "kv_footprint_realized": True,
        "blocks_layer0": nb,
        "residual_tokens_layer0": res_used,
        "dense_window_tokens": dl,
    }


def quant_config(quant: str):
    import torch
    from transformers import BitsAndBytesConfig
    q = (quant or "").strip().lower()
    if q in ("nf4", "int4", "4bit", "4-bit", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=("fp4" if q == "fp4" else "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ), torch.bfloat16
    if q in ("int8", "8bit", "8-bit"):
        return BitsAndBytesConfig(load_in_8bit=True), torch.float16
    return None, torch.float16


def load_plain(model_id: str, quant: str, eager: bool):
    """Plain HF model for the dense control and the baseline arms."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    qc, dtype = quant_config(quant)
    kw: Dict[str, Any] = {"device_map": "cuda", "dtype": dtype}
    if qc is not None:
        kw["quantization_config"] = qc
    if eager:
        # snapkv / h2o read real attention weights; SDPA does not return them.
        kw["attn_implementation"] = "eager"
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        # Text models shipped only inside a multimodal wrapper (Ministral-3).
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model


def derive_stop_ids(tok) -> set:
    out = set()
    for t in (tok.eos_token_id, getattr(tok, "pad_token_id", None)):
        if isinstance(t, int):
            out.add(t)
        elif isinstance(t, list):
            out.update(x for x in t if isinstance(x, int))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_files(paths: List[str]) -> None:
    metrics = official_metrics()
    by_run: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[str, Dict[str, Any]] = {}
    sysm: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for p in paths:
        store = ResumableJSONL(p, config=None, strict_config=False)
        # One record per key, last write wins, so a successful retry supersedes
        # the failed attempt before it and neither is double-counted.
        latest = store.load_latest()
        store.close()
        recs = [r for r in latest.values() if not r.get("error")]
        n_err = len(latest) - len(recs)
        if n_err:
            print(f"[warn] {os.path.basename(p)}: {n_err} item(s) still failing; "
                  f"excluded from the score. Rerun to retry them.")
        if not recs:
            continue
        mp = p + ".meta.json"
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                meta[p] = json.load(f)
        for r in recs:
            ds = r.get("dataset")
            if ds not in metrics:
                continue
            pred = post_process(r.get("text", ""), ds)
            best = 0.0
            for gt in r.get("answers", []):
                best = max(best, metrics[ds](pred, gt,
                                             all_classes=r.get("all_classes")))
            by_run[p][ds].append(best)
            for k in ("ttft_s", "decode_tps", "peak_decode_gb", "kv_physical_gb",
                      "kv_compression_x", "prompt_tokens"):
                if isinstance(r.get(k), (int, float)):
                    sysm[p][k].append(float(r[k]))

    for p in sorted(by_run):
        m = meta.get(p, {})
        tag = (f"{m.get('model','?')} arm={m.get('arm','?')} "
               f"preset={m.get('preset','-')} quant={m.get('quant','-')} "
               f"maxlen={m.get('max_length','-')}")
        print(f"\n=== {os.path.basename(p)} ===\n{tag}")
        print(f"{'task':>22} {'n':>4} {'score':>8}")
        allv = []
        for ds in sorted(by_run[p]):
            v = by_run[p][ds]
            allv += v
            print(f"{ds:>22} {len(v):>4} {100*sum(v)/len(v):>8.2f}")
        if allv:
            print(f"{'MACRO(all samples)':>22} {len(allv):>4} "
                  f"{100*sum(allv)/len(allv):>8.2f}")
        s = sysm.get(p, {})
        if s:
            def avg(k):
                return sum(s[k]) / len(s[k]) if s.get(k) else float("nan")
            print(f"  ttft {avg('ttft_s'):.2f}s | decode {avg('decode_tps'):.1f} tok/s "
                  f"| peak {avg('peak_decode_gb'):.2f} GB "
                  f"| KV {avg('kv_physical_gb'):.3f} GB "
                  f"({avg('kv_compression_x'):.1f}x) "
                  f"| ctx {avg('prompt_tokens'):.0f} tok")


# ─────────────────────────────────────────────────────────────────────────────

def _paired_bootstrap(deltas: List[float], iters: int = 10000,
                      seed: int = 1234) -> tuple:
    """95% CI for the mean of paired per-item differences.

    Paired, because every arm answers the SAME LongBench items: bootstrapping
    the difference removes item difficulty from the variance entirely. Two
    independent per-arm intervals at n=120 would overlap for gaps that are in
    fact consistent across almost every item, and the reader would call a real
    effect inconclusive.
    """
    import random
    if not deltas:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def compare_files(paths: List[str]) -> None:
    """One table, arms as rows — the layout the paper needs.

    `--score` prints a table per file, which answers "how did this arm do" but
    not "which arm won", and comparing nine of them by eye across a scrollback
    is how a wrong row ends up in a paper. This puts every arm on one grid at
    the same tasks, and refuses to average over a task an arm is missing:
    a macro computed over a different denominator per row is not a ranking.
    """
    metrics = official_metrics()
    rows: Dict[str, Dict[str, List[float]]] = {}
    by_item: Dict[str, Dict[str, float]] = {}
    meta_by_arm: Dict[str, Dict[str, Any]] = {}
    sysm: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for p in sorted(paths):
        store = ResumableJSONL(p, config=None, strict_config=False)
        latest = store.load_latest()
        store.close()
        recs = [r for r in latest.values() if not r.get("error")]
        if not recs:
            continue
        meta = {}
        if os.path.exists(p + ".meta.json"):
            with open(p + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        arm = meta.get("arm", "?")
        if meta.get("preset") and arm == "dkv":
            arm = f"dkv/{meta['preset']}"
        meta_by_arm[arm] = meta
        by_item.setdefault(arm, {})
        per: Dict[str, List[float]] = defaultdict(list)
        for r in recs:
            ds = r.get("dataset")
            if ds not in metrics:
                continue
            pred = post_process(r.get("text", ""), ds)
            best = 0.0
            for gt in r.get("answers", []):
                best = max(best, metrics[ds](pred, gt, all_classes=r.get("all_classes")))
            per[ds].append(best)
            # Keyed per item, so arms can be compared on the items they share.
            by_item.setdefault(arm, {})[r.get("key", f"{ds}#{r.get('idx')}")] = best
            for k in ("ttft_s", "decode_tps", "e2e_tps", "peak_decode_gb",
                      "kv_physical_gb", "kv_compression_x"):
                if isinstance(r.get(k), (int, float)):
                    sysm[arm][k].append(float(r[k]))
        rows[arm] = per

    if not rows:
        print("no scoreable results")
        return

    tasks = sorted({t for per in rows.values() for t in per})
    m0 = next(iter(meta_by_arm.values()), {})
    print(f"\n=== LongBench — {m0.get('model','?')} "
          f"@ max_length {m0.get('max_length','?')}, quant {m0.get('quant','?')}, "
          f"thinking {m0.get('thinking', '?')} ===")
    hdr = f"{'arm':>14} " + " ".join(f"{t[:12]:>13}" for t in tasks) + f" {'MACRO':>7} {'KV GB':>7} {'cmp x':>6}"
    print(hdr)
    print("-" * len(hdr))

    def _order(a):
        return {"dense": 0}.get(a, 1), a
    for arm in sorted(rows, key=_order):
        per = rows[arm]
        cells, allv, complete = [], [], True
        for t in tasks:
            v = per.get(t)
            if v:
                cells.append(f"{100*sum(v)/len(v):>13.2f}")
                allv += v
            else:
                cells.append(f"{'-':>13}")
                complete = False
        macro = f"{100*sum(allv)/len(allv):>7.2f}" if (allv and complete) else f"{'n/a':>7}"
        s = sysm.get(arm, {})
        kv = (sum(s["kv_physical_gb"]) / len(s["kv_physical_gb"])) if s.get("kv_physical_gb") else float("nan")
        cx = (sum(s["kv_compression_x"]) / len(s["kv_compression_x"])) if s.get("kv_compression_x") else float("nan")
        print(f"{arm:>14} " + " ".join(cells) + f" {macro} {kv:>7.3f} {cx:>6.1f}")

    missing = [a for a, per in rows.items() if len(per) < len(tasks)]
    if missing:
        print(f"\nMACRO withheld for {', '.join(sorted(missing))}: not every task "
              f"is present, and averaging over a different denominator per row "
              f"would not be a ranking. Finish those arms, or compare per task.")

    # ── paired comparison against dense ─────────────────────────────────────
    if "dense" in by_item and len(by_item) > 1:
        print(f"\nPaired against dense, on the items both arms answered "
              f"(95% CI, 10k bootstrap resamples over items):")
        print(f"{'arm':>14} {'n':>5} {'mean delta':>11} {'95% CI':>20}  verdict")
        for arm in sorted(a for a in by_item if a != "dense"):
            shared = sorted(set(by_item[arm]) & set(by_item["dense"]))
            if not shared:
                continue
            deltas = [100 * (by_item[arm][k] - by_item["dense"][k]) for k in shared]
            mean = sum(deltas) / len(deltas)
            lo, hi = _paired_bootstrap(deltas)
            # "resolved" only when the interval excludes zero. At n=120 with
            # this much per-item variance, plenty of gaps will not be.
            verdict = ("worse than dense" if hi < 0 else
                       "better than dense" if lo > 0 else
                       "NOT RESOLVED at this n")
            print(f"{arm:>14} {len(shared):>5} {mean:>11.2f} "
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>20}  {verdict}")
        print("\nPaired because every arm answers the SAME items: bootstrapping\n"
              "the per-item difference takes item difficulty out of the variance.\n"
              "Two independent per-arm intervals would overlap for gaps that are\n"
              "in fact consistent across nearly every item.")


def main():
    # Before ANY torch.compile: without cl.exe on PATH the Inductor
    # decode path falls back to eager and every latency number here
    # understates DKV. Quality is unaffected; timings are not.
    from msvc_env import ensure_msvc
    global MSVC_OK
    MSVC_OK = ensure_msvc()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-4.2-8b")
    ap.add_argument("--arm", default="dkv",
                    help="dense | dkv | streamingllm | snapkv | h2o | kivi2 | "
                         "kivi4 | int8_kv")
    ap.add_argument("--preset", default="mid",
                    choices=["low", "mid", "high", "ultra"])
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--max-length", type=int, default=15500,
                    help="context budget; prompts longer than this are "
                         "middle-truncated, as the official harness does")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--out", default="")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--baseline-params", default="{}",
                    help="JSON of method params, e.g. '{\"budget\": 2048}'")
    ap.add_argument("--thinking", action="store_true",
                    help="let reasoning models emit their <think> block. OFF by "
                         "default: the per-task generation budgets are far too "
                         "small to hold chain-of-thought AND an answer.")
    ap.add_argument("--score", nargs="*", default=None,
                    help="score existing JSONL result files and exit")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="one table with every arm as a row, and exit")
    args = ap.parse_args()

    if args.compare is not None:
        paths: List[str] = []
        for pat in (args.compare or ["paper/results/longbench/*.jsonl"]):
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            raise SystemExit("no result files matched")
        return compare_files(paths)

    if args.score is not None:
        paths: List[str] = []
        for pat in (args.score or ["paper/results/longbench/*.jsonl"]):
            paths.extend(sorted(glob.glob(pat)))
        if not paths:
            raise SystemExit("no result files matched")
        return score_files(paths)

    out = args.out or os.path.join(
        REPO, "paper", "results", "longbench",
        f"{args.model.split('/')[-1]}_{args.arm}_{args.preset}_{args.quant}.jsonl")

    cfg = {"model": args.model, "arm": args.arm, "quant": args.quant,
           "max_length": args.max_length, "num_samples": args.num_samples,
           "preset": args.preset if args.arm == "dkv" else None,
           "baseline_params": json.loads(args.baseline_params),
           "thinking": bool(args.thinking),
           "decode_defaults": "serving" if args.arm == "dkv" else None,
           # The attention path the PREFILL ran under. Recorded because
           # snapkv/h2o used to force a whole-model eager load, which made
           # their prefill ~5x slower than every arm they are compared with.
           # Rows from before that fix must not merge with rows after it.
           "prefill_attn": "sdpa",
           # Invalidates DKV rows when the decode ARITHMETIC changes. The
           # config guard alone cannot see a kernel fix: the attention-scale
           # correction left this dict byte-identical, so a resume appended
           # post-fix rows to 81 pre-fix ones and produced a table where
           # gov_report had moved 10.63 -> 28.55 while hotpotqa sat at exactly
           # its old 15.12. Baseline arms do not run this code, so they are
           # not fingerprinted and are not needlessly discarded.
           "dkv_decode_rev": (decode_fingerprint() if args.arm == "dkv" else None),
           "protocol": "longbench-official-v1"}
    store = ResumableJSONL(out, config=cfg)
    done = store.load_done()
    print(f"[ckpt] {out}\n[ckpt] {len(done)} items already recorded")

    prompts, maxlens = official_config()
    work = []
    for ds in args.datasets:
        if ds not in prompts:
            raise SystemExit(f"unknown LongBench task: {ds}")
        for i, row in enumerate(load_task(ds, args.num_samples)):
            key = f"{ds}#{i}"
            if key not in done:
                work.append((key, ds, i, row))
    print(f"[ckpt] {len(work)} items pending")
    if not work:
        print("nothing to do; scoring what is on disk")
        store.close()
        return score_files([out])

    sys.path.insert(0, ACTIVE)
    import torch
    import kv_baselines as KB

    # ── load the arm ──
    if args.arm == "dkv":
        os.environ.setdefault("DKV_RSVD_SEED", "1234")
        os.environ.setdefault("DKV_SVD_SEED", "1234")
        cwd = os.getcwd()
        os.chdir(ACTIVE)
        # BENCHMARK THE CONFIGURATION THAT SHIPS, NOT THE LIBRARY DEFAULTS.
        # serving/cli.py and the OpenAI gateway both call this before building
        # the wrapper; a harness that constructs DKVHFWrapper directly does not,
        # and the difference is not cosmetic. With DKV_SPARSE_BIAS unset it
        # takes the value "0.0", which selects the COMBINED decode branch --
        # a code path production never runs. dkv_attention.py's own comment on
        # that branch records what has lived there: "It only ever fired on the
        # combined branch -- DKV_SPARSE_BIAS unset or 0.0, the LIBRARY DEFAULT.
        # BEST_DECODE_DEFAULTS sets it to auto, which takes the production
        # branch, so everything going through the serving defaults was
        # unaffected and never saw it", measured at KL 11.76 with top-1 0/5
        # against KL 0.00125 and 5/5 once fixed.
        #
        # Every explicit env still wins (this is setdefault), so the arm knobs
        # set above are untouched.
        from serving.decode_config import apply_best_decode_defaults
        apply_best_decode_defaults()
        from serving.hf_dkv_wrapper import DKVHFWrapper
        w = DKVHFWrapper(model_id=args.model,
                         config={"preset": args.preset,
                                 "quantization": args.quant or None})
        w.ensure_loaded()
        tok = w.tokenizer
        os.chdir(cwd)
        model = None
    else:
        tok, model = load_plain(args.model, args.quant, KB.needs_eager(args.arm))
        w = None
    stop_ids = derive_stop_ids(tok)
    bparams = json.loads(args.baseline_params)

    t_start = time.time()
    for n, (key, ds, i, row) in enumerate(work, 1):
        gen_len = maxlens[ds]
        prompt = build_prompt(tok, row, ds, prompts[ds], args.max_length,
                              thinking=args.thinking)
        ntok = len(tok(prompt).input_ids)
        try:
            if args.arm == "dkv":
                sid = f"lb-{ds}-{i}"
                try:
                    w.clear_session(sid)
                except Exception:                                # noqa: BLE001
                    pass
                w.active_session = sid
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                # NO query_text: the dense and baseline arms never see the
                # question ahead of the context, so handing it to DKV's router
                # would give this arm information the others do not have.
                text = w.generate(prompt, max_new_tokens=gen_len,
                                  temperature=0.0, top_p=1.0,
                                  repetition_penalty=1.0)
                if isinstance(text, dict):
                    text = text.get("text", "")
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                text, clean_strip = strip_prompt_echo(text, prompt, tok,
                                                      budget_tokens=gen_len)
                try:
                    sess = w.manager.sessions.get(sid) or {}
                    blocks = int(sum(sess.get("num_blocks") or []))
                except Exception:                                # noqa: BLE001
                    blocks = -1
                rec = {"text": text, "wall_s": wall, "blocks": blocks,
                       "prompt_echo_clean": clean_strip,
                       # END-TO-END, not decode. The wrapper's generate() does
                       # prefill and decode in one call, so this wall time
                       # contains both and is NOT comparable to the baselines'
                       # decode_tps (which excludes prefill by construction).
                       # Kept under a different name on purpose; the decode
                       # throughput comparison belongs to the dedicated
                       # systems harness, which separates the two phases.
                       "e2e_tps": (gen_len / wall) if wall > 0 else 0.0,
                       "peak_decode_gb": (torch.cuda.max_memory_allocated() / 1e9
                                          if torch.cuda.is_available() else 0.0)}
                try:
                    rec.update(dkv_kv_bytes(w.manager, ntok, sid))
                except Exception as e:                           # noqa: BLE001
                    # Never let the memory accounting take down a quality run;
                    # record why it is missing instead of leaving a silent hole.
                    rec["kv_accounting_error"] = f"{type(e).__name__}: {e}"
                try:
                    w.clear_session(sid)
                except Exception:                                # noqa: BLE001
                    pass
            else:
                ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
                rec = KB.run_baseline(model, tok, ids, args.arm, "cuda",
                                      gen_len, stop_ids, args.chunk, bparams)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [{n}/{len(work)}] {key}: ERROR {type(e).__name__}: {str(e)[:160]}")
            store.append(key, dataset=ds, idx=i, error=f"{type(e).__name__}: {e}",
                         prompt_tokens=ntok)
            continue

        # `rec` wins on any shared field: run_baseline measures prompt_tokens
        # from the ids it actually fed the model, which is the authoritative
        # count if the two ever disagree.
        fields = {"dataset": ds, "idx": i, "prompt_tokens": ntok,
                  "inductor_fused": MSVC_OK,
                  "answers": row.get("answers", []),
                  "all_classes": row.get("all_classes"),
                  "length": row.get("length")}
        fields.update(rec)
        store.append(key, **fields)
        el = time.time() - t_start
        print(f"  [{n}/{len(work)}] {key} {ntok} tok -> "
              f"{repr(rec.get('text','')[:60])} "
              f"({el/n:.1f}s/item, ~{(len(work)-n)*el/n/60:.0f} min left)",
              flush=True)

    store.close()
    print()
    score_files([out])


if __name__ == "__main__":
    main()
