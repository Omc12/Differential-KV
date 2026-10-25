@echo off
rem V7 -- fill the RULER curves Figure 1 leaves empty.
rem   p1  dense read through DKV's tiled path at 131k on Qwen3.5-4B, the
rem       V3_BEYOND items (5 per task): a dense reference where the paper's
rem       main comparison sits (preallocated dense spills there; tiled fits).
rem   p2  the same on granite-4.2-8b at 24k (10 per task): dense past its
rem       preallocated ceiling, beside streaming DKV and eviction.
rem   p3  DKV/mid on granite at 4k-16k (20 per task), the lengths dense was
rem       scored at, prefilled exactly as the system does there.
rem RULER on the paper's prompt (no answer prefix). Every phase is resumable:
rem re-running a phase skips the items already in its store.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=0

if not "%1"=="" goto %1

:p1
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense_tiled --min-length 131072 --max-length 131072 --per-task 5 --baseline-params "{}" --out %R%\ruler\V7_Qwen3.5-4B_dense_tiled_131k.jsonl
echo V7 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dense_tiled --min-length 24576 --max-length 24576 --per-task 10 --baseline-params "{}" --out %R%\ruler\V7_granite_dense_tiled_24k.jsonl
echo V7 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
setlocal
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 4096 --max-length 16384 --per-task 20 --out %R%\ruler\V7_granite_dkv_mid_4k-16k.jsonl
endlocal
echo V7 P3 COMPLETE
