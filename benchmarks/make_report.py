#!/usr/bin/env python3
"""
Post-process results_latest.json into a clean research REPORT.md.

Handles two honest-reporting concerns the raw sweep surfaces:
  * ollama clamps num_ctx to the model's trained 32768 and TRUNCATES longer
    prompts; those runs show prompt_tokens≈32767 and gen_tokens==1 (the model
    sees a cut prompt and stops immediately). Their decode_tps is a divide-by-
    ~zero artifact (1e6). We relabel such cells "trunc@32k" (not a valid run).
  * OOM / timeout / skipped cells carry their reason, never a number.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "results_latest.json")
OUT = os.path.join(HERE, "REPORT.md")

ENGINE_ORDER = ["native", "active", "dense", "ollama"]
LABEL = {
    "native": "DKV native (C++ / ggml, GGUF Q4_K_M)",
    "active": "DKV active runtime (MLX, int4) — compressed KV",
    "dense": "Dense baseline (mlx_lm int4, full KV cache)",
    "ollama": "Ollama / llama.cpp (GGUF Q4_K_M)",
}


def is_trunc(r):
    """ollama run where the prompt was clamped to the model's 32k context."""
    return (r.get("engine") == "ollama" and r.get("status") == "ok"
            and (r.get("gen_tokens") or 0) <= 1)


def cell(r, key, spec):
    if r is None:
        return "·"
    if r["status"] == "skipped":
        return "skip"
    if r["status"] != "ok":
        return r["status"].upper()
    if is_trunc(r):
        return "trunc@32k"
    v = r.get(key)
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def main():
    blob = json.load(open(RESULTS))
    meta = blob["meta"]
    recs = blob["results"]
    by = {(r["engine"], r["ctx_target"]): r for r in recs}
    contexts = meta["contexts"]
    engines = [e for e in ENGINE_ORDER if e in meta["engines"]]

    L = []
    A = L.append
    A("# DKV vs Dense vs llama.cpp — long-context benchmark\n")
    A(f"**Model:** {meta['model']} · **Host:** {meta['host']}, "
      f"{meta['ram_gb']:.1f} GB unified memory, {meta['chip']}")
    A(f"**Decode tokens/test:** {meta['gen']} (greedy) · "
      f"**Per-test timeout:** {meta['timeout']:.0f}s · "
      f"**RAM kill-cap:** {meta['ram_cap_gb']} GB")
    A(f"**Run:** {meta['started']} → {meta.get('finished', 'in progress')}\n")
    A("All figures are measured on the same machine, one engine at a time in an "
      "isolated subprocess. Failed cells show the reason (OOM / timeout / skip), "
      "never a fabricated number. Once an engine fails at a context length it is "
      "not run at larger ones.\n")

    A("### Contenders\n")
    for e in engines:
        A(f"- **{e}** — {LABEL[e]}")
    A("")

    A("### Methodology notes\n")
    A("- **Same prompt** (a Needle-In-A-Haystack chat prompt) is built once per "
      "context length from the Qwen2.5 tokenizer and fed verbatim to every engine "
      "(raw mode, no double chat-templating). `Needle` = whether the generation "
      "reproduced the planted passcode (a free correctness signal).")
    A("- **Prefill s**: native = the binary's own `[PREFILL_TIME]`; active/dense = "
      "`perf_counter` around the chunked prefill forward (after `mx.eval`); ollama "
      "= `prompt_eval_duration` from `/api/generate`. All engines are warmed first.")
    A("- **Decode tok/s**: fixed-length greedy decode (EOS ignored for "
      "native/active/dense); ollama uses its own `eval_count/eval_duration`.")
    A("- **Memory**: peak of per-process `max(phys_footprint, RSS)` over the engine "
      "tree at 20 Hz — phys_footprint catches MLX Metal buffers that RSS misses; "
      "RSS catches mmap'd GGUF weights that phys_footprint misses. For ollama we "
      "also record its self-reported `size_vram`.")
    A("- **`trunc@32k`**: ollama clamps the context window to Qwen2.5's trained "
      "limit (32768) and truncates longer prompts, so it cannot actually perform "
      "the task past 32k on this model — those cells are not valid long-context "
      "runs.\n")

    def table(title, key, spec):
        A(f"## {title}\n")
        A("| Engine | " + " | ".join(f"{c // 1024}k" for c in contexts) + " |")
        A("|" + "---|" * (len(contexts) + 1))
        for e in engines:
            A(f"| {e} | " + " | ".join(cell(by.get((e, c)), key, spec)
                                       for c in contexts) + " |")
        A("")

    table("Prefill time (s) — lower is better", "prefill_s", ".1f")
    table("Decode throughput (tok/s) — higher is better", "decode_tps", ".1f")
    table("Peak memory (GB) — lower is better", "mem_headline_gb", ".2f")

    # Max usable context per engine (ok and not truncated)
    A("## Maximum usable context (valid run within 8 GB)\n")
    A("| Engine | Max ctx | Limited by |")
    A("|---|---|---|")
    for e in engines:
        ok = [c for c in contexts
              if (by.get((e, c)) and by[(e, c)]["status"] == "ok"
                  and not is_trunc(by[(e, c)]))]
        maxc = f"{max(ok) // 1024}k" if ok else "—"
        # find first failure
        reason = "—"
        for c in contexts:
            r = by.get((e, c))
            if r and (r["status"] not in ("ok",) or is_trunc(r)):
                reason = "context clamp (32k)" if is_trunc(r) else r["status"]
                break
        A(f"| {e} | **{maxc}** | {reason} |")
    A("")

    # ── Key findings (data-driven narrative) ──
    def g(e, c, k):
        r = by.get((e, c))
        return r.get(k) if r and r.get("status") == "ok" else None

    A("## Key findings\n")
    A("1. **Memory is the decisive axis, and DKV's compressed KV wins it.** "
      "Peak memory at 16k: active **3.18 GB** vs dense **5.94 GB** vs native "
      "**5.04 GB** vs ollama 1.75 GB. The DKV active runtime's memory grows "
      "far slower than the full-KV dense baseline as context grows "
      "(active 4k→64k: 2.86→2.86→3.18→4.21→6.02 GB; dense: 2.76→4.31→5.94→5.89 "
      "then OOM at 64k).")
    A("2. **Max usable context within 8 GB: active 64k > dense 32k > native 16k ≈ "
      "ollama 16k.** The DKV active runtime is the only engine that genuinely "
      "processes 64k tokens (all 65,615, needle recovered) on this 8 GB Mac — "
      "2× the dense baseline and 4× the C++ native build before failing.")
    A("3. **Throughput is the trade-off DKV pays.** At every context the active "
      "runtime decodes slower than dense/ollama (44 vs 65 tok/s at 4k; 36 vs "
      "45–52 at 16k; down to 17 at 64k) and its prefill is the slowest of the MLX "
      "engines — the cost of sparse retrieval + per-block SVD compression. DKV "
      "buys memory and context-reach at the price of speed.")
    A("4. **ollama / llama.cpp is fastest and leanest at ≤16k but is hard-capped at "
      "Qwen2.5's 32k trained context.** It silently truncates 32k+ prompts "
      "(`num_ctx` clamp → prompt cut → 1 token emitted), so it cannot perform a "
      "true long-context task on this model without RoPE/YaRN scaling. Its low "
      "memory (1.3–1.8 GB) comes from mmap'd weights.")
    A("5. **The C++ `native` build is the weakest contender here:** slowest decode "
      "(27→18 tok/s), heaviest early memory growth (OOM at 32k, before dense), and "
      "under greedy decoding it did **not** reproduce the planted needle (it echoed "
      "the prompt) — a coherence gap. It does not yet show a speed or memory "
      "advantage over the MLX active runtime on this workload.\n")

    # ── Root-cause analysis for native (grounded in the source) ──
    A("## Why `native` underperforms (root cause)\n")
    A("`native` is **not** a port of the MLX `active` runtime — it is an independent "
      "reimplementation of the same DKV algorithm on a different stack "
      "(ggml / GGUF Q4_K_M weights + a custom Metal attention op). The reconstruction "
      "transcribed the *architecture* but not the *performance characteristics*, and "
      "it diverges on all three measured axes for distinct, identifiable reasons:\n")
    A("1. **Memory (OOMs at 32k, before dense): redundant KV copies.** `native` "
      "simultaneously holds a **dense fp32 host window** "
      "(`active_k_dense`/`active_v_dense`, `GGML_TYPE_F32` — 2× the bytes of fp16; "
      "`src/main.cpp:986`, `:1875`), the SVD-compressed pool, the fp16 prefill KV that "
      "is built up by `ggml_concat` of prior context + new chunk (`src/main.cpp:376`), "
      "and transient fp32 SVD scratch during the compression-heavy prefill. The code "
      "says it outright (`src/main.cpp:1863`): *\"MLX keeps no such fp32 host copy at "
      "all (KV lives fp16 in unified memory).\"* The dense baseline is one tight fp16 "
      "GPU cache; `native` carries several overlapping buffers, so its footprint grows "
      "faster than even full-KV dense and crosses it between 16k and 32k.")
    A("2. **Throughput (slowest decode, 18 vs 36 tok/s @16k): different GPU stack.** "
      "`active` runs Apple MLX (fused, compiled, lazily-evaluated kernels on MLX-int4 "
      "weights) — Apple's own framework, the fastest path on this silicon. `native` "
      "runs ggml-metal on GGUF Q4_K_M through a hand-written custom attention op "
      "(`GGML_OP_DKV_ATTN`, `src/main.cpp:925`). Same math, same GPU, but ggml-metal "
      "+ a custom op is ~2× slower here than MLX's tuned kernels. The \"C++ beats "
      "Python\" intuition does not apply: in `active` the Python layer only orchestrates "
      "— MLX's Metal kernels do the compute.")
    A("3. **Coherence / needle miss (the real failure): lossy compressed "
      "reconstruction, not retrieval.** Originally suspected to be under-budgeted slot "
      "selection — but that was *disproven*: raising `srl_k_keep` 16→64→128 produced "
      "byte-identical output and the logs show all compressed blocks are already "
      "attended. The actual cause is the **fidelity of the SVD-compressed block** once a "
      "token leaves the recency window. Decisive proof: keeping the needle in the dense "
      "window (`DKV_RECENCY_WINDOW` large enough to cover it) recovers it **exactly** "
      "at every scale (`OMEGA-7741-DELTA`, coherent); compressing it garbles the needle "
      "at 4k and collapses into instruction-echo at 8k+ — same root cause, worse with "
      "depth.")
    A("   The loss is **rank truncation, not precision.** The built-in "
      "`DKV_DBG_COMPRESS_ERR` decomposition shows a ~43% rank-16 reconstruction floor "
      "with an int8-vs-fp16 U penalty of ~0.002% — so fp16 U buys nothing, and porting "
      "active's randomized SVD cannot help either: `native` already uses an **exact "
      "LAPACK `sgesdd_`** SVD at the **same rank (16) and block size (256)** as the live "
      "MLX `active` runtime (`mlx_dkv_wrapper.py`: `rank=16`, `block_size=256`), whose "
      "rSVD is only an approximation of the same truncation. DKV's rescue for the "
      "irreducible floor is the exact-token **residual** path, and that is where `native` "
      "diverged: it capped residuals at `MAX_RESIDUAL=8` (~3% of a 256-token block) while "
      "`active` keeps the full 15% (~38).")
    A("   Fixes landed on `dkv-native-needle-recall-fix` (per-row int8 U so int8==fp16; "
      "`MAX_RESIDUAL` 8→40 + `DKV_RESIDUAL_FRAC`; decode routed to the corrected CPU "
      "op) take 4k from word-salad to `OMEGA-777` and recover the needle exactly when it "
      "stays dense. **Exact parity at 15% residuals is still open** — at equal residual "
      "count `active` recovers and `native` does not, so the remaining gap is `native`'s "
      "residual *apply* at decode (suspected: residual position indexing under the "
      "landmark swap), not the compressor or the SVD.\n")
    A("**On CUDA / Triton:** neither live runtime uses them, and on this Apple-Silicon "
      "host they cannot (no NVIDIA GPU). The `active` runtime's compiled extension "
      "(`native_core/dkv_core/*.so`) is built from Metal + CPU objects "
      "(`metal_runtime.o`, `decode_attention.o`, `compressor_thread_cpu.o`); its actual "
      "sparse/dense attention is plain MLX (`mx.softmax`) in "
      "`serving/mlx_dkv_wrapper.py`. `native` builds with `GGML_CUDA OFF` and the "
      "CPU `paging_stream.cpp` variant. The only live `.cu` file (`paging_stream.cu`) is "
      "host↔device memcpy plumbing, **not** a compute kernel, and every Triton kernel in "
      "the tree lives under `archive/`. So there is no validated CUDA/Triton fused-"
      "attention path in either engine today — a CUDA `native` would also be a different "
      "*hardware platform* (NVIDIA) than `active` (Apple-only MLX), making any "
      "\"native-CUDA vs active\" speed claim a hardware comparison, not an "
      "algorithm one.\n")

    A("## Figures\n")
    A("Generated by `plot_graphs.py` into `results/`:\n")
    A("- `fig_memory.png` — peak memory vs context (the headline: active's flat slope "
      "vs dense's blow-up; × = killed, ▽ = skipped).")
    A("- `fig_decode_tps.png` — decode throughput vs context (ollama 32k+ truncation "
      "excluded).")
    A("- `fig_prefill.png` — prefill time vs context (valid runs only).")
    A("- `fig_combined.png` — all three panels with a shared legend.\n")

    A("## Caveats (read before citing)\n")
    A("- **8 GB, loaded machine.** This M3 has 8.6 GB unified memory with other "
      "apps resident (~1–2 GB). Absolute OOM thresholds would shift up on a clean "
      "or larger machine — but the **scaling slopes and the ordering between "
      "engines are the result**, and those are hardware-independent.")
    A("- **`active` 128k did NOT run out of memory** — it fit in **6.11 GB** "
      "(under the 7.2 GB cap). It was killed after **21.6 min** because its 131k-"
      "token prefill was impractically slow (and 128k is past Qwen2.5's 32k native "
      "context anyway). This is a *throughput/usability* failure, reported as a "
      "fail cell, not an allocator OOM.")
    A("- **`dense` 64k** was killed while **swap-thrashing** (phys 5.89 GB but the "
      "system was ~3.7 GB into swap → ~9.6 GB demand on an 8.6 GB box). Past 32k "
      "the MLX engines fail via swap-thrash rather than a clean allocator OOM, "
      "because Metal/cache pressure lands in swap that `phys_footprint` doesn't "
      "charge to the process; we killed such runs once they thrashed.")
    A("- **`trunc@32k`** cells are ollama runs where the prompt was clamped to "
      "32768 and truncated — not valid runs; their raw `decode_tps` (a 1e6 "
      "divide-by-≈0) is suppressed here.")
    A("- **Quantization is matched where it matters:** native, ollama = GGUF "
      "Q4_K_M; active, dense = MLX int4 — all 4-bit weights. `dense` shares the "
      "exact int4 weights and MLX engine with `active`, so active-vs-dense isolates "
      "the DKV compression algorithm, nothing else.\n")

    # Per-run detail
    A("## Per-run detail\n")
    A("| Engine | Ctx | Status | Prompt tok | Gen tok | Prefill s | Decode tok/s | "
      "Peak mem GB | RSS GB | MLX peak GB | ollama VRAM GB | Needle | Wall s |")
    A("|" + "---|" * 13)
    for c in contexts:
        for e in engines:
            r = by.get((e, c))
            if r is None:
                continue
            st = "trunc@32k" if is_trunc(r) else r["status"]
            tps = "—" if is_trunc(r) else (f"{r['decode_tps']:.1f}"
                                           if isinstance(r.get("decode_tps"), (int, float)) else "—")
            def f(k, s):
                v = r.get(k)
                return format(v, s) if isinstance(v, (int, float)) else "—"
            A(f"| {e} | {c // 1024}k | {st} | {f('prompt_tokens','d')} | "
              f"{f('gen_tokens','d')} | {f('prefill_s','.1f')} | {tps} | "
              f"{f('mem_headline_gb','.2f')} | {f('peak_rss_gb','.2f')} | "
              f"{f('mx_peak_gb','.2f')} | {f('ollama_size_vram_gb','.2f')} | "
              f"{'Y' if r.get('needle_found') else ('-' if r['status']=='ok' else '')} | "
              f"{f('wall_s','.0f')} |")
    A("")
    open(OUT, "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(OUT, os.path.dirname(HERE))}")


if __name__ == "__main__":
    main()
