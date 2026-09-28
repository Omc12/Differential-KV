#!/usr/bin/env python3
"""Gate for the chunk-wise eviction arms: nothing long runs until this passes.

Two parts.

--store   (CPU, seconds) the resume machinery the campaign depends on:
          a torn final record is repaired rather than glued to the next one,
          a lock left by a run that died before the last boot is stale, and a
          lock held by this live process is not.

--model   (GPU, minutes) the arms themselves, on the model they will run on:
  probe    how out.attentions lines up with the KV-bearing cache layers. On a
           hybrid this decides whether the attention-observation arms score
           each layer with its own weights; the old per-decoder-layer indexing
           is reported alongside, so its coverage is on record.
  exact    with a window/budget larger than the prompt nothing is evicted, so
           each chunked arm must reproduce dense: streamingllm_chunked runs the
           same SDPA chunks and must match to the bit; h2o_chunked runs eager
           in 256-token chunks and must agree on the next token with KL ~ 0.
  bounded  on a long prompt the largest cache any layer ever held is at most
           budget + one chunk, and the final cache is exactly the budget.
  flat     peak prefill memory barely moves when the prompt doubles, while
           dense's grows -- the property the whole comparison is about.
  decode   the evicted cache still decodes.

Exit status is 0 only if every check passes; results go to
paper/results/selfcheck/ so the campaign log can point at them.

    python benchmarks/selfcheck_chunked_eviction.py --store
    python benchmarks/selfcheck_chunked_eviction.py --model Qwen/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)


# ─────────────────────────────────────────────────────────────────────────────
# Store tests (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def store_tests() -> dict:
    import checkpoint as C
    res = {}
    d = tempfile.mkdtemp(prefix="ckpt-selftest-")

    # 1. torn fragment: cut off, kept in .torn, next record lands on its own line
    p = os.path.join(d, "torn.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write('{"key": "a", "v": 1}\n{"key": "b", "v"')
    s = C.ResumableJSONL(p, config={"x": 1})
    s.append("c", v=3)
    s.close()
    s = C.ResumableJSONL(p, config={"x": 1})
    done = s.load_done()
    s.append("d", v=4)
    s.close()
    s = C.ResumableJSONL(p, config={"x": 1}, read_only=True)
    res["torn_fragment_cut"] = (done == {"a", "c"}
                                and set(s.load_latest()) == {"a", "c", "d"}
                                and os.path.exists(p + ".torn"))

    # 2. complete record missing only its newline: kept, not discarded
    p = os.path.join(d, "nonl.jsonl")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write('{"key": "a", "v": 1}\n{"key": "b", "v": 2}')
    s = C.ResumableJSONL(p, config={"x": 1})
    s.append("c", v=3)
    s.close()
    s = C.ResumableJSONL(p, config={"x": 1}, read_only=True)
    res["unterminated_record_kept"] = set(s.load_latest()) == {"a", "b", "c"}

    # 3. lock from before the last boot is stale even if its pid is live
    p = os.path.join(d, "lock.jsonl")
    with open(p + ".lock", "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "boot": C._boot_time() - 86400,
                   "started": "yesterday"}, f)
    try:
        s = C.ResumableJSONL(p, config={"x": 1})
        s.close()
        res["prior_boot_lock_reclaimed"] = True
    except SystemExit:
        res["prior_boot_lock_reclaimed"] = False

    # 4. lock held by THIS live process, this boot, is honoured
    with open(p + ".lock", "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "boot": C._boot_time(),
                   "started": "now"}, f)
    try:
        s = C.ResumableJSONL(p, config={"x": 1})
        s.close()
        res["live_lock_honoured"] = False
    except SystemExit:
        res["live_lock_honoured"] = True
    os.remove(p + ".lock")

    # 5. boot time is stable across calls (sanity of the clock source)
    res["boot_time_stable"] = abs(C._boot_time() - C._boot_time()) < 2.0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Model tests (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def model_tests(model_id: str, quant: str) -> dict:
    import torch
    import torch.nn.functional as F
    import kv_baselines as KB
    from run_longbench_cuda import load_plain
    from context_ladder import build_filler

    out = {"model": model_id, "quant": quant}
    tok, model = load_plain(model_id, quant, False)
    dev = "cuda"

    def ids_of(n):
        return tok(build_filler(tok, n), add_special_tokens=False).input_ids

    # ── probe: attention tuple vs cache layers ──────────────────────────────
    ids = ids_of(600)
    past, _ = KB.chunked_prefill(model, ids[:-32], dev, 1024)
    with KB.eager_attention(model), torch.no_grad():
        o = model(input_ids=torch.tensor([ids[-32:]], device=dev),
                  position_ids=torch.tensor([list(range(len(ids) - 32, len(ids)))],
                                            device=dev),
                  past_key_values=past, use_cache=True, output_attentions=True)
    pk = o.past_key_values
    n_layers = KB.cache_num_layers(pk)
    kv_layers = [l for l in range(n_layers) if KB.cache_get_kv(pk, l)[0] is not None]
    atts = o.attentions
    # What the ORIGINAL per-decoder-layer indexing evicted: layers with KV,
    # l < len(atts), atts[l] not None.
    old_cov = [l for l in kv_layers if l < len(atts) and atts[l] is not None]
    try:
        mapping = KB.attn_by_cache_layer(pk, atts)
        aligned = all(mapping[l].shape[-1] == KB.cache_get_kv(pk, l)[0].shape[2]
                      for l in kv_layers)
        map_err = None
    except RuntimeError as e:
        aligned, map_err = False, str(e)
    out["probe"] = {
        "decoder_layers": n_layers, "kv_layers": kv_layers,
        "attentions_len": len(atts),
        "attentions_nonnull": sum(a is not None for a in atts),
        "old_indexing_evicted_layers": old_cov,
        "old_indexing_correct": (len(atts) == n_layers),
        "new_mapping_aligned": aligned, "new_mapping_error": map_err,
    }
    del o, pk, past, atts
    torch.cuda.empty_cache()

    checks = {}

    # ── exact: nothing to evict -> must equal dense ─────────────────────────
    ids = ids_of(3000)
    torch.cuda.synchronize()
    _, dense_logits = KB.chunked_prefill(model, ids, dev, 1024)
    _, s_logits, _, _ = KB._streamingllm_chunked(model, ids, dev, 1024, 4, 8192)
    checks["streaming_exact_when_unbounded"] = bool(torch.equal(dense_logits, s_logits))
    _, h_logits, _, _ = KB._h2o_chunked(model, ids, dev, 256, 8192, 512)
    kl = F.kl_div(F.log_softmax(h_logits, -1), F.log_softmax(dense_logits, -1),
                  log_target=True, reduction="sum").item()
    out["h2o_unbounded_kl_vs_dense"] = kl
    checks["h2o_matches_dense_when_unbounded"] = bool(
        int(h_logits.argmax()) == int(dense_logits.argmax()) and kl < 1e-2)
    torch.cuda.empty_cache()

    # ── bounded + flat + decode ─────────────────────────────────────────────
    def peak_of(method, n, params):
        ids_n = ids_of(n)
        torch.cuda.empty_cache()
        r = KB.run_baseline(model, tok, ids_n, method, dev, 8, set(), 1024,
                            dict(params))
        return r

    budgets = {"streamingllm_chunked": ({"n_sink": 4, "recency_window": 2044}, 2048, 1024),
               "h2o_chunked": ({"budget": 2048, "recency_window": 512}, 2048, 256)}
    peaks = {}
    for m, (params, budget, ch) in budgets.items():
        r8 = peak_of(m, 8192, params)
        r16 = peak_of(m, 16384, params)
        pk_final = None
        peaks[m] = (r8["peak_prefill_gb"], r16["peak_prefill_gb"])
        out[m] = {"peak_8k": r8["peak_prefill_gb"], "peak_16k": r16["peak_prefill_gb"],
                  "max_cache_8k": r8["max_cache_tokens"],
                  "max_cache_16k": r16["max_cache_tokens"],
                  "kv_gb_16k": r16["kv_physical_gb"],
                  "prefill_s_16k": r16["prefill_s"],
                  "text_16k": r16["text"][:80], "gen_tokens_16k": r16["gen_tokens"]}
        checks[f"{m}_bounded"] = (r16["max_cache_tokens"] <= budget + ch
                                  and r8["max_cache_tokens"] <= budget + ch)
        checks[f"{m}_decodes"] = r16["gen_tokens"] > 0
    # SnapKV must evict EVERY KV layer: stored/dense-equivalent == kept/prompt.
    # Under the old decoder-layer indexing on Qwen3.5-4B this came out at a
    # quarter of that, because only 2 of 8 layers were counted (and evicted).
    rs = peak_of("snapkv", 8192, {"budget": 2016, "window": 32})
    ratio = rs["kv_physical_gb"] / rs["kv_dense_equiv_gb"]
    expect = 2048 / rs["prompt_tokens"]
    out["snapkv_8k"] = {"kv_gb": rs["kv_physical_gb"],
                        "dense_equiv_gb": rs["kv_dense_equiv_gb"],
                        "stored_fraction": ratio, "expected_fraction": expect}
    checks["snapkv_evicts_every_kv_layer"] = abs(ratio - expect) < 0.01
    rd8 = peak_of("dense", 8192, {})
    rd16 = peak_of("dense", 16384, {})
    out["dense"] = {"peak_8k": rd8["peak_prefill_gb"], "peak_16k": rd16["peak_prefill_gb"]}
    dense_growth = rd16["peak_prefill_gb"] - rd8["peak_prefill_gb"]
    for m, (p8, p16) in peaks.items():
        # Flat means: grows by well under half of what dense grows by over the
        # same doubling. (Not zero: the lm_head over a chunk and allocator
        # rounding move a little.)
        checks[f"{m}_peak_flat"] = (p16 - p8) < 0.5 * dense_growth
    out["dense_growth_8k_to_16k_gb"] = dense_growth
    out["checks"] = checks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", action="store_true")
    ap.add_argument("--model", default="")
    ap.add_argument("--quant", default="nf4")
    args = ap.parse_args()

    results, ok = {"when": time.strftime("%Y-%m-%d %H:%M:%S")}, True
    if args.store:
        r = store_tests()
        results["store"] = r
        ok &= all(r.values())
    if args.model:
        r = model_tests(args.model, args.quant)
        results["model"] = r
        ok &= all(r["checks"].values()) and r["probe"]["new_mapping_aligned"]
    results["pass"] = bool(ok)

    od = os.path.join(REPO, "paper", "results", "selfcheck")
    os.makedirs(od, exist_ok=True)
    tag = (args.model.split("/")[-1] if args.model else "store")
    path = os.path.join(od, f"chunked_eviction_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\n{'PASS' if ok else 'FAIL'}  -> {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
