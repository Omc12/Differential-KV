@echo off
rem Streaming-prefill ceilings after the router fix (7c5dc99f), which removed a
rem whole-pool residual dequantization from every decode step. Three residual
rem budgets on granite, and Qwen past its old 131k. New stores; resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ladder
set DKV_STREAMING_COMPRESS=1
set DKV_PREFILL_LOWMEM=1
set DKV_REMAT_CACHE=0
set DKV_TRITON_STRICT=1

"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --preset mid --contexts 24576 32768 49152 65536 98304 --timeout 3600 --out %R%\FIX_granite_mid_res128_streamlow.jsonl
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --preset high --contexts 24576 32768 49152 65536 98304 --timeout 3600 --out %R%\FIX_granite_high_res256_streamlow.jsonl
set DKV_MAX_RESIDUAL_TOKENS=512
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --preset high --contexts 24576 32768 49152 65536 --timeout 3600 --out %R%\FIX_granite_high_res512_streamlow.jsonl
set DKV_MAX_RESIDUAL_TOKENS=
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --preset mid --contexts 131072 163840 196608 --timeout 3600 --out %R%\FIX_Qwen3.5-4B_mid_res128_streamlow.jsonl

echo ROUTERFIX LADDERS COMPLETE
