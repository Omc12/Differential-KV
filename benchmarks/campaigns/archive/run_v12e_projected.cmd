@echo off
rem V12 g3 / p6 -- exact window = auto budget minus the PROJECTED final store
rem (whole prompt compressed), fixed per prompt. p5 used the store so far, so on
rem granite (store outgrows the budget) the window closed only as memory ran out
rem and the live-memory guard fired at 68674 tokens. The guard now warns only when
rem it changes a decision. Same items and environment as p1/p4/p5.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set DKV_ROUTE_TOPP_STATS=1
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304

if not "%1"=="" goto %1

:g3
"%PY%" benchmarks\run_ruler_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset mid --min-length 114688 --max-length 114688 --tasks niah_single_1 --per-task 1 --out %R%\ruler\V12_granite_dkv_mid_auto3_115k_reach.jsonl
echo V12 G3 COMPLETE
if not "%1"=="" goto :eof

:p6
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 3 --out %R%\ruler\V12_Qwen3.5-4B_dkv_mid_resid_ada_auto3_131k.jsonl
echo V12 P6 COMPLETE
