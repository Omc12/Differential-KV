@echo off
rem V12 p3 -- the adaptive block count's own effect on the residual router.
rem Identical to V12 p1 (memory window, 3 per task) except DKV_ROUTE_ADAPTIVE=0,
rem so p1 - p3 is the adaptive rule alone on the current code. (V12 p1 tied V10's
rem fixed-K store, but V10 ran older decode code, so that pair does not isolate it.)
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
set DKV_STREAM_ELASTIC_POLICY=memory
set DKV_ROUTE_ADAPTIVE=0
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 3 --out %R%\ruler\V12_Qwen3.5-4B_dkv_mid_resid_fixedk_memwin_131k.jsonl
echo V12 P3 COMPLETE
