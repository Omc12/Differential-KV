@echo off
rem V3 PART S -- the SYSTEMS table (MLSys): TTFT, decode tok/s, peak allocated +
rem reserved, KV bytes, vs context, every arm through ONE harness and ONE
rem definition (bench_systems_cuda.py, protocol systems-v2: TTFT and decode rate
rem from inside a single call; median of 3 repeats after a discarded warmup;
rem min/max kept; spill = allocated OR reserved > 94% of the card).
rem Batch size 1, NF4 weights, 12.9 GB card, Windows; CUDA graphs and Inductor
rem as shipped (the harness refuses to run without cl.exe).
rem One file per arm configuration; resumable. Est. ~7-9 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\systems
set QW=Qwen/Qwen3.5-4B
set GR=ibm-granite/granite-4.2-8b
set G=64
set NOP={}

rem ===== Qwen3.5-4B =====
rem baselines (each arm stops at its first spill). "dense" = HF cache with the
rem allocator fix (defrag), as the paper's dense; "dense_static" = preallocated.
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dense --gen %G% --contexts 8192 16384 32768 65536 --baseline-params "{\"defrag\": true}" --out %R%\V3_Qwen3.5-4B_dense.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dense --gen %G% --contexts 8192 16384 32768 65536 98304 122880 --baseline-params "{\"static\": true}" --out %R%\V3_Qwen3.5-4B_dense_static.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms snapkv --gen %G% --contexts 8192 16384 32768 49152 --baseline-params "{\"budget\": 2016, \"window\": 32}" --out %R%\V3_Qwen3.5-4B_snapkv.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms kivi4_tiled --gen %G% --contexts 8192 16384 32768 65536 131072 --baseline-params "%NOP%" --out %R%\V3_Qwen3.5-4B_kivi4_tiled.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms streamingllm_chunked h2o_chunked --gen %G% --contexts 32768 131072 --baseline-params "%NOP%" --out %R%\V3_Qwen3.5-4B_evict_chunked.jsonl
rem DKV, low-rank store: exact (lean, default) to its ceiling; streaming beyond it
setlocal
set DKV_STREAMING_COMPRESS=0
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --gen %G% --contexts 8192 16384 32768 65536 131072 --out %R%\V3_Qwen3.5-4B_dkv_exact.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --gen %G% --contexts 196608 262144 --out %R%\V3_Qwen3.5-4B_dkv_auto.jsonl
endlocal
rem DKV, hybrid store
setlocal
set DKV_STORE=hybrid
set DKV_STREAMING_COMPRESS=0
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --gen %G% --contexts 8192 16384 32768 65536 131072 --out %R%\V3_Qwen3.5-4B_hybrid_exact.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --gen %G% --contexts 196608 262144 --out %R%\V3_Qwen3.5-4B_hybrid_auto.jsonl
endlocal

rem ===== granite-4.2-8b =====
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dense --gen %G% --contexts 4096 8192 16384 --baseline-params "{\"defrag\": true}" --out %R%\V3_granite_dense.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dense --gen %G% --contexts 4096 8192 16384 20480 --baseline-params "{\"static\": true}" --out %R%\V3_granite_dense_static.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms snapkv --gen %G% --contexts 4096 8192 16384 --baseline-params "{\"budget\": 2016, \"window\": 32}" --out %R%\V3_granite_snapkv.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms kivi4_tiled --gen %G% --contexts 4096 8192 16384 32768 65536 98304 --baseline-params "%NOP%" --out %R%\V3_granite_kivi4_tiled.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms streamingllm_chunked h2o_chunked --gen %G% --contexts 16384 65536 --baseline-params "%NOP%" --out %R%\V3_granite_evict_chunked.jsonl
setlocal
set DKV_STREAMING_COMPRESS=0
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --gen %G% --contexts 4096 8192 16384 24576 --out %R%\V3_granite_dkv_exact.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=24576
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --gen %G% --contexts 32768 65536 131072 --out %R%\V3_granite_dkv_auto.jsonl
endlocal
setlocal
set DKV_STORE=hybrid
set DKV_STREAMING_COMPRESS=0
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --gen %G% --contexts 4096 8192 16384 24576 --out %R%\V3_granite_hybrid_exact.jsonl
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=24576
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --gen %G% --contexts 32768 65536 131072 --out %R%\V3_granite_hybrid_auto.jsonl
endlocal

echo V3 PART S COMPLETE
