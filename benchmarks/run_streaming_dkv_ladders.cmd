@echo off
rem DKV with compression DURING prefill (DKV_IMMEDIATE_PREFILL_COMPRESS=1) and
rem no remat cache. By default DKV keeps every block exact until the prompt
rem ends and compresses afterwards, so its prefill peak carries the whole
rem uncompressed KV -- measured on granite as ingest_chunk growing at dense's
rem rate. These ladders measure the streaming mode. Resumable per point.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set OUT=paper\results\ladder
set DKV_IMMEDIATE_PREFILL_COMPRESS=1
set DKV_REMAT_CACHE=0

"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --contexts 16384 24576 32768 49152 65536 98304 131072 --out %OUT%\granite-4.2-8b_mid_nf4_dkv_streaming.jsonl

"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --contexts 65536 98304 131072 196608 --out %OUT%\Qwen3.5-4B_mid_nf4_dkv_streaming.jsonl

echo STREAMING DKV LADDERS COMPLETE
