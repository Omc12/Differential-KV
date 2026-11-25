@echo off
rem V12 -- residual router back as the default, adaptive block count on top of it.
rem V11 found the key-box router (default from 6db9743a) cost 22 points at 131k
rem against V10 on the same items; a bisect put it on the box (DKV_ROUTE_BOX=0
rem answered 2/2 niah_multiquery items, =1 answered 0/2). Same harness, items and
rem environment as V10 p1 / V11 (RULER with the answer prefix, Qwen3.5-4B 131k,
rem DKV/mid streaming). Every phase is resumable.
rem   p1  defaults + DKV_STREAM_ELASTIC_POLICY=memory, 3 per task: pairs with
rem       V10P noquery (same window rule, fixed-K residual router)
rem   p2  defaults as shipped (token window), 5 per task: pairs with V11 p1/p2
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set DKV_ROUTE_TOPP_STATS=1
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set L=--min-length 131072 --max-length 131072

if not "%1"=="" goto %1

:p1
setlocal
set DKV_STREAM_ELASTIC_POLICY=memory
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %L% --per-task 3 --out %R%\ruler\V12_Qwen3.5-4B_dkv_mid_resid_ada_memwin_131k.jsonl
endlocal
echo V12 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %L% --per-task 5 --out %R%\ruler\V12_Qwen3.5-4B_dkv_mid_resid_ada_131k.jsonl
echo V12 P2 COMPLETE
