@echo off
rem V12 p5 -- auto budget sized from the session's own store plus a slack, so the
rem live-memory guard stays out of normal runs. V12 p4 answers depended on what ran
rem earlier in the process: the guard reads live memory, and a 70 MB difference
rem moved one layer's compression by a chunk at ~125k. Same items and environment
rem as p1/p4. Pairs with p1 (memory window) and p4 (auto, guard-driven).
rem   g2  granite-4.2-8b reach at 114688 under the same rule.
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

:p5
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 3 --out %R%\ruler\V12_Qwen3.5-4B_dkv_mid_resid_ada_auto2_131k.jsonl
echo V12 P5 COMPLETE
if not "%1"=="" goto :eof

:g2
"%PY%" benchmarks\run_ruler_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset mid --min-length 114688 --max-length 114688 --tasks niah_single_1 --per-task 1 --out %R%\ruler\V12_granite_dkv_mid_auto2_115k_reach.jsonl
echo V12 G2 COMPLETE
