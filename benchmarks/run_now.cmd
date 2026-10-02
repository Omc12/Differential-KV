@echo off
rem NOW: does block tiling change quality, and what is left of the transient?
rem 1. forced-streaming RULER at 32k on Qwen3.5-4B, same 130 items as the
rem    untiled run, paired against it.
rem 2. allocator trace at 32k and 64k with tiling on.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_TRITON_STRICT=1
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 32768 --max-length 32768 --per-task 10 --out %R%\ruler\Qwen3.5-4B_dkv_mid_forcedstream_tiled_32k.jsonl
set DIAG_DEPTH=3
"%PY%" benchmarks\diag_granite_snapshot.py --out %R%\diag\qwen_stream_trace_tiled.jsonl --points "Qwen/Qwen3.5-4B|dkv|32768" "Qwen/Qwen3.5-4B|dkv|65536"
set DKV_STREAMING_COMPRESS=
echo NOW COMPLETE
