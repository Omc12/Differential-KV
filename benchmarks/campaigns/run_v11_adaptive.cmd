@echo off
rem V11 -- adaptive block count (DKV_ROUTE_ADAPTIVE) past dense's limit.
rem Same harness, items and environment as V10 p1 (RULER with the answer
rem prefix, Qwen3.5-4B at 131,072, DKV/mid streaming), first 5 items per task.
rem p1 and p2 run the SAME code and differ only in the router's block count,
rem so p1 - p2 is the adaptive rule alone. Every phase is resumable.
rem   p1  Qwen3.5-4B 131k  DKV/mid, adaptive K (the 2026-11-24 default)
rem   p2  Qwen3.5-4B 131k  DKV/mid, fixed K=16 box router (DKV_ROUTE_ADAPTIVE=0)
rem   p3  granite 65k      DKV, codes store, 3 items (reach check after the
rem                        decode-buffer cuts)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set DKV_ROUTE_TOPP_STATS=1
set Q=--min-length 131072 --max-length 131072 --per-task 5

if "%1"=="resume" goto resume
if not "%1"=="" goto %1

:p1
setlocal
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q% --out %R%\ruler\V11_Qwen3.5-4B_dkv_mid_adaptive_131k.jsonl
endlocal
echo V11 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
setlocal
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set DKV_ROUTE_ADAPTIVE=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q% --out %R%\ruler\V11_Qwen3.5-4B_dkv_mid_fixedk_131k.jsonl
endlocal
echo V11 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
setlocal
set DKV_STORE=codes
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 65536 --max-length 65536 --per-task 1 --tasks niah_single_1 niah_multikey_2 qa_1 --out %R%\ruler\V11_granite_dkv_codes_65k.jsonl
endlocal
echo V11 P3 COMPLETE
if not "%1"=="" goto :eof
goto :eof

:resume
call "%~f0" p1
call "%~f0" p2
call "%~f0" p3
