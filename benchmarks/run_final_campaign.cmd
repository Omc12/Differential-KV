@echo off
rem FINAL CAMPAIGN -- only what is NOT already measured.
rem
rem The system under test: DKV_STREAMING_COMPRESS=auto. A prompt within the exact
rem limit takes the default DKV path (byte-identical, verified), so every default
rem DKV result already in paper\results is this system's result for that range
rem and is not re-run. A longer prompt streams, with the streaming fixes switched
rem on per prompt. What is new is therefore everything BEYOND the limits:
rem   granite-4.2-8b  limit 16,384  -> 24,576 and 32,768
rem   Qwen3.5-4B      limit 98,304  -> 131,072
rem plus dense at 65,536 (it fits there with the allocator fix, so it must be
rem scored) and the systems rows that predate the SnapKV alignment fix.
rem
rem Resumable: after a power cut run this file again; every harness skips what
rem is on disk. DKV_TRITON_STRICT=1: a kernel failure raises instead of falling
rem back, so no number here can come from the slow path.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set SL={\"n_sink\": 4, \"recency_window\": 2044}
set H2={\"budget\": 2048, \"recency_window\": 512, \"prefill_chunk\": 256}
set DKV_TRITON_STRICT=1

rem ── 1. granite beyond its exact ceiling: RULER 24k + 32k, 10 items per task ──
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm streamingllm_chunked --min-length 24576 --max-length 32768 --per-task 10 --baseline-params "%SL%" --out %R%\ruler\BEYOND_granite_streamingllm_chunked_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm h2o_chunked --min-length 24576 --max-length 32768 --per-task 10 --baseline-params "%H2%" --out %R%\ruler\BEYOND_granite_h2o_chunked_24k-32k.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 24576 --max-length 32768 --per-task 10 --out %R%\ruler\BEYOND_granite_dkv_mid_auto_24k-32k.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

rem ── 2. Qwen beyond its exact ceiling: RULER 131k, 5 items per task ──────────
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked --min-length 131072 --max-length 131072 --per-task 5 --baseline-params "%SL%" --out %R%\ruler\BEYOND_Qwen3.5-4B_streamingllm_chunked_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm h2o_chunked --min-length 131072 --max-length 131072 --per-task 5 --baseline-params "%H2%" --out %R%\ruler\BEYOND_Qwen3.5-4B_h2o_chunked_131k.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\BEYOND_Qwen3.5-4B_dkv_mid_auto_131k.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

rem ── 3. dense at 65,536 with the allocator fix (all 260 items) ───────────────
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense --min-length 65536 --max-length 65536 --baseline-params "{\"defrag\": true}" --out %R%\ruler\Qwen3.5-4B_dense_defrag_max65536.jsonl

rem ── 4. systems rows: SnapKV after the alignment fix; DKV in streaming mode ───
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms snapkv --contexts 8192 16384 32768 49152 --out %R%\systems\FINAL_Qwen3.5-4B_snapkv_aligned.jsonl
"%PY%" benchmarks\bench_multiquery_cuda.py --model %QW% --arms snapkv --contexts 16384 32768 --pattern append --out %R%\multiquery\FINAL_Qwen3.5-4B_snapkv_aligned_append.jsonl
"%PY%" benchmarks\bench_multiquery_cuda.py --model %QW% --arms snapkv --contexts 16384 32768 --pattern independent --out %R%\multiquery\FINAL_Qwen3.5-4B_snapkv_aligned_independent.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --contexts 131072 --out %R%\systems\FINAL_Qwen3.5-4B_dkv_auto_streaming.jsonl
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --contexts 24576 32768 --out %R%\systems\FINAL_granite_dkv_auto_streaming.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=

echo FINAL CAMPAIGN COMPLETE
