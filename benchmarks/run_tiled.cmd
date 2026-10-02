@echo off
rem STREAMING DKV WITH BLOCK-TILED HISTORY ATTENTION (DKV_PREFILL_BLOCK_TILE=16,
rem the new default): where do its ceilings move? 128-token answers, as every
rem other ceiling in the paper. Then a trace at 64k to confirm the history-
rem attention transient no longer scales with history. Resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_TRITON_STRICT=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --gen 128 --contexts 131072 147456 163840 196608 229376 262144 --out %R%\ladder\TILED_Qwen3.5-4B_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 45056 49152 57344 65536 81920 98304 --out %R%\ladder\TILED_granite_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=
set DKV_STREAMING_COMPRESS=1
set DIAG_DEPTH=3
"%PY%" benchmarks\diag_granite_snapshot.py --out %R%\diag\qwen_stream_trace_tiled.jsonl --points "Qwen/Qwen3.5-4B|dkv|32768" "Qwen/Qwen3.5-4B|dkv|65536"
set DKV_STREAMING_COMPRESS=
echo TILED COMPLETE
