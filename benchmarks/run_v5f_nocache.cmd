@echo off
rem V5-F -- granite codes store with the decode rematerialization cache held off (DKV_REMAT_RESERVE_GB=30: no layer is ever kept, every step rebuilds).
rem V5-E (fixed window) still spilled at 65,536 with a normal first token (131 s) and
rem decode at 0.53 tok/s: the overflow is at decode, where code payloads go through
rem the memory-sized rematerialization cache. ~25 min.
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
set DKV_REMAT_RESERVE_GB=30
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 98304 --timeout 5400 --out paper\results\ladder\V5_granite_codes4_nocache_auto.jsonl
echo V5F COMPLETE
