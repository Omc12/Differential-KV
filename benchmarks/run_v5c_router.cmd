@echo off
rem V5-C -- is the codes store's 131k gap to tiled KIVI-4 the router? The same
rem codes (DKV_STORE=codes, 4-bit) attending every block, on the three tasks
rem where V5 p1 trails tiled KIVI-4 (lookalike multi-key 2 and 3, multi-query),
rem items 0-4 as p1. Pairs item for item with p1 and with tiled KIVI-4. ~40 min.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set DKV_STORE=codes
set DKV_QUANT_BITS=4
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set RULER_ANSWER_PREFIX=0
set DKV_TOPK_BLOCKS=0
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 131072 --max-length 131072 --tasks niah_multikey_2 niah_multikey_3 niah_multiquery --per-task 5 --out paper\results\ruler\V5_Qwen3.5-4B_codes4_attendall_131k.jsonl
echo V5C COMPLETE
