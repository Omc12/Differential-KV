@echo off
rem V9 -- does DKV's store need the question? The low-rank store orders its
rem residual rows by the prompt's last 64 tokens; DKV_RESIDUAL_NO_QUERY=1 turns
rem that look-ahead off. V6 tested it once (RULER, Qwen3.5-4B, 131k). Here, on
rem granite-4.2-8b:
rem   p1  LongBench at 12k, DKV/mid with the look-ahead, current code (control)
rem   p2  LongBench at 12k, DKV/mid without it (same 120 items)
rem   p3  RULER at 4k-16k, DKV/mid without it, 10 per task, paired with the V7
rem       store (same env, look-ahead on)
rem Every phase is resumable: re-running a phase skips items already stored.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\Appdata\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=0
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384

if not "%1"=="" goto %1

:p1
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm dkv --preset mid --max-length 12000 --num-samples 20 --out %R%\longbench\V9_granite_dkv_mid_query_len12000.jsonl
echo V9 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
setlocal
set DKV_RESIDUAL_NO_QUERY=1
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm dkv --preset mid --max-length 12000 --num-samples 20 --out %R%\longbench\V9_granite_dkv_mid_noquery_len12000.jsonl
endlocal
echo V9 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
setlocal
set DKV_RESIDUAL_NO_QUERY=1
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 4096 --max-length 16384 --per-task 10 --out %R%\ruler\V9_granite_dkv_mid_noquery_4k-16k.jsonl
endlocal
echo V9 P3 COMPLETE
