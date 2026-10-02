@echo off
rem OVERNIGHT, in order; resumable after a power cut.
rem 1. where streaming DKV with block tiling stops: granite past 49k, Qwen past 262k
rem 2. the paused chunk-wise KIVI-4 campaign (quality beyond the limit, LongBench)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_TRITON_STRICT=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 45056 49152 57344 65536 81920 98304 131072 --out %R%\ladder\TILED_granite_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --gen 128 --contexts 131072 147456 163840 196608 229376 262144 327680 393216 524288 --out %R%\ladder\TILED_Qwen3.5-4B_dkv_auto.jsonl
set DKV_STREAMING_COMPRESS=
set DKV_STREAM_AUTO_TOKENS=
call benchmarks\run_kivi.cmd
echo OVERNIGHT COMPLETE
