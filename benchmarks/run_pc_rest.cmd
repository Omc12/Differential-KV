@echo off
rem THE LAST PC RUNS. Resumable: rerun after a power cut.
rem 1. granite dense RULER to 16k: the earlier run stopped after 46 of 260
rem    items at 16k; the paper's dense reference for granite needs all of them.
rem 2. streaming vs exact DKV on RULER at one length (Qwen3.5-4B 32k, 10 items
rem    per task, paired against the exact-mode items with the same indices).
rem 3. the paused ceiling bisection.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_TRITON_STRICT=1

"%PY%" benchmarks\run_ruler_cuda.py --model ibm-granite/granite-4.2-8b --arm dense --max-length 16384 --out %R%\ruler\granite-4.2-8b_dense_max16384.jsonl

set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 32768 --max-length 32768 --per-task 10 --out %R%\ruler\Qwen3.5-4B_dkv_mid_forcedstream_32k.jsonl
set DKV_STREAMING_COMPRESS=

call benchmarks\run_bisect.cmd

echo PC REST COMPLETE
