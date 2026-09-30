@echo off
rem Four ladders that decide whether DKV's ceiling claim survives a fair dense
rem baseline and a DKV without its remat cache. Each ladder is resumable per
rem point: after a power cut, run this file again.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set OUT=paper\results\ladder

rem 1. Qwen dense with the allocator defragmented after every chunk
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dense --contexts 49152 65536 98304 131072 --baseline-params "{\"defrag\": true}" --out %OUT%\Qwen3.5-4B_mid_nf4_dense_defrag.jsonl

rem 2. granite dense, same fix (expected to stay KV-bound)
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dense --contexts 16384 24576 32768 --baseline-params "{\"defrag\": true}" --out %OUT%\granite-4.2-8b_mid_nf4_dense_defrag.jsonl

rem 3. granite DKV without the remat cache
set DKV_REMAT_CACHE=0
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --contexts 16384 24576 32768 49152 65536 98304 --out %OUT%\granite-4.2-8b_mid_nf4_dkv_noremat.jsonl

rem 4. Qwen DKV without the remat cache
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --contexts 65536 98304 131072 --out %OUT%\Qwen3.5-4B_mid_nf4_dkv_noremat.jsonl
set DKV_REMAT_CACHE=

echo FAIRNESS LADDERS COMPLETE
