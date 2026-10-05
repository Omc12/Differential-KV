@echo off
rem V3 PART A -- DKV streaming under the 2026-10-04 defaults (fused history
rem attention E1 and the elastic window D2 on). Re-measures only what those
rem defaults made stale: streaming timing and streaming quality. DKV exact,
rem streaming peaks and reach are unchanged and are NOT re-run.
rem
rem The switches are set explicitly (they are the defaults anyway) so every
rem result's config records them and can never merge with a pre-V3 store.
rem Resumable: run again after a power cut; finished points/items are skipped.
rem Old-path numbers this replaces (est. now ~2.5x faster): ~8 h then, ~4-5 h now.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1

rem -- A1. ceilings + timing (same contexts / streaming switch as the paper's TILED ladders)
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 24576 32768 49152 65536 98304 131072 --timeout 7200 --out %R%\ladder\V3_granite_dkv_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 131072 163840 196608 229376 262144 --timeout 7200 --out %R%\ladder\V3_Qwen3.5-4B_dkv_auto.jsonl
endlocal

rem -- A2. quality, forced streaming at lengths exact mode would take (cost of streaming)
setlocal
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm dkv --preset mid --max-length 12000 --num-samples 20 --out %R%\longbench\V3_granite-4.2-8b_dkv_mid_stream_len12000.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 32768 --max-length 32768 --per-task 10 --out %R%\ruler\V3_Qwen3.5-4B_dkv_mid_forcedstream_32k.jsonl
endlocal

rem -- A3. quality beyond the exact limits (the system as shipped: auto)
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 24576 --max-length 32768 --per-task 10 --out %R%\ruler\V3_BEYOND_granite_dkv_mid_auto_24k-32k.jsonl
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\V3_BEYOND_Qwen3.5-4B_dkv_mid_auto_131k.jsonl
endlocal

echo V3 PART A COMPLETE
