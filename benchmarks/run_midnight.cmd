@echo off
rem MIDNIGHT CAMPAIGN -- the two checks the paper's ceiling claim still needs.
rem
rem 1. A preallocated dense baseline (transformers StaticCache). The 2x claim
rem    was measured against dense with the allocator's cache returned per chunk,
rem    which the paper calls a lower bound. This is the strongest dense the
rem    framework offers. Gated: it must reproduce dense's continuation.
rem 2. Ceilings at a realistic answer length. Every ladder so far generated 8
rem    tokens; at 64k dense then paged on 93 of 260 RULER items, the ones with
rem    long answers. Every arm's deciding rungs are re-measured at 128 tokens.
rem 3. Concurrency with the decode-time memory fixes on, since the paper's
rem    current concurrency row (DKV 7 sessions vs dense 11) predates them.
rem
rem Resumable: after a power cut run this file again; every harness skips what
rem is on disk. DKV_TRITON_STRICT=1 so no number comes from a fallback path.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set ST={\"static\": true}
set DF={\"defrag\": true}
set DKV_TRITON_STRICT=1

rem ── 0. gate: StaticCache dense must answer exactly as dense does ───────────
"%PY%" benchmarks\selfcheck_static_dense.py
if errorlevel 1 (
  echo STATIC DENSE SELFCHECK FAILED -- stopping before any ladder
  exit /b 1
)

rem ── 1. preallocated dense ladders, 128-token answers ───────────────────────
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 49152 65536 98304 131072 --baseline-params "%ST%" --out %R%\ladder\GEN128_Qwen3.5-4B_dense_static.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 16384 24576 32768 49152 --baseline-params "%ST%" --out %R%\ladder\GEN128_granite_dense_static.jsonl

rem ── 2. the deciding rungs again at 128-token answers ───────────────────────
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 49152 65536 98304 --baseline-params "%DF%" --out %R%\ladder\GEN128_Qwen3.5-4B_dense_defrag.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 16384 24576 --baseline-params "%DF%" --out %R%\ladder\GEN128_granite_dense_defrag.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 98304 131072 163840 --out %R%\ladder\GEN128_Qwen3.5-4B_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 16384 32768 49152 --out %R%\ladder\GEN128_granite_dkv_auto.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

rem ── 3. sessions resident at 16k: DKV with the decode-memory fixes ──────────
set DKV_REMAT_CACHE=0
"%PY%" benchmarks\bench_concurrency_cuda.py --model %QW% --arms dkv --ctx 16384 --out %R%\concurrency\Qwen3.5-4B_ctx16384_dkv_noremat.jsonl
set DKV_STREAMING_COMPRESS=1
set DKV_PREFILL_LOWMEM=1
set DKV_ROUTER_SLOT_DEQUANT=1
set DKV_CLAMP_DECODE_RANK=1
"%PY%" benchmarks\bench_concurrency_cuda.py --model %QW% --arms dkv --ctx 16384 --out %R%\concurrency\Qwen3.5-4B_ctx16384_dkv_streamprofile.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_PREFILL_LOWMEM=
set DKV_ROUTER_SLOT_DEQUANT=
set DKV_CLAMP_DECODE_RANK=
set DKV_REMAT_CACHE=

rem ── 4. preallocated dense latency, three repetitions ───────────────────────
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dense --contexts 32768 65536 --baseline-params "%ST%" --out %R%\systems\MIDNIGHT_Qwen3.5-4B_dense_static.jsonl

rem ── 5. fill the paper's empty cells ───────────────────────────────────────
rem SnapKV on granite was never laddered (ceiling table) or timed at its ceiling.
"%PY%" benchmarks\context_ladder.py --model %GR% --arms snapkv --contexts 16384 24576 32768 --out %R%\ladder\granite-4.2-8b_mid_nf4_snapkv_aligned.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms snapkv --gen 128 --contexts 16384 24576 32768 --out %R%\ladder\GEN128_granite_snapkv_aligned.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms snapkv --contexts 16384 24576 --out %R%\systems\MIDNIGHT_granite_snapkv_aligned.jsonl
rem Streaming DKV timed at a length every arm serves (forced; the system would
rem prefill these exactly), so the latency table compares like with like.
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --contexts 32768 --out %R%\systems\MIDNIGHT_Qwen3.5-4B_dkv_forced_stream_32k.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --contexts 16384 --out %R%\systems\MIDNIGHT_granite_dkv_forced_stream_16k.jsonl
set DKV_STREAMING_COMPRESS=
rem DKV/high beyond its exact limit (65,536 on Qwen3.5-4B): does it fit at 131k,
rem and what does it score there. The RULER run is the long one (~5.5 h).
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=65536
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --preset high --contexts 131072 --out %R%\ladder\Qwen3.5-4B_high_nf4_dkv_auto.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset high --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\BEYOND_Qwen3.5-4B_dkv_high_auto_131k.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

echo MIDNIGHT CAMPAIGN COMPLETE
