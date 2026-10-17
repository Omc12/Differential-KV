@echo off
rem Trace allocated memory per chunk and around the boundary compression, Qwen3.5-4B
rem auto streaming at 131k with elastic on (spilled at 18.58 GB in V3 part A).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set DKV_DBG_ELASTIC=1
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --gen 8 --contexts 131072 --timeout 3600 --out paper\results\diag\TRACE_qwen_elastic_131k.jsonl
echo TRACE COMPLETE
