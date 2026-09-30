@echo off
rem TRACK A (experimental, opt-in): DKV with compression DURING prefill and the
rem low-memory history attention. Every result goes to a *_streamlow* file so
rem nothing here touches the default-DKV results. Resumable: after a power cut
rem run this file again; finished points/items are skipped.
rem
rem DKV_TRITON_STRICT=1: a Triton kernel failure RAISES instead of silently
rem falling back to the PyTorch decoder, so no number here can come from the
rem fallback. (Results before commit 996ed334 did, and read garbage ranks; they
rem are in paper\results\superseded.)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set DKV_STREAMING_COMPRESS=1
set DKV_PREFILL_LOWMEM=1
set DKV_REMAT_CACHE=0
set DKV_TRITON_STRICT=1

rem 1. Ceilings
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --contexts 16384 24576 32768 49152 65536 98304 131072 --timeout 7200 --out %R%\ladder\granite-4.2-8b_mid_nf4_dkv_streamlow.jsonl
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --contexts 32768 65536 98304 131072 --timeout 7200 --out %R%\ladder\Qwen3.5-4B_mid_nf4_dkv_streamlow.jsonl

rem 2. Quality: LongBench granite @12k, same config as the committed dkv/mid arm
"%PY%" benchmarks\run_longbench_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset mid --max-length 12000 --num-samples 20 --out %R%\longbench\granite-4.2-8b_dkv_mid_streamlow_len12000.jsonl

rem 3. Quality: RULER Qwen3.5-4B 4k-32k, then 64k
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --max-length 32768 --out %R%\ruler\Qwen3.5-4B_dkv_mid_streamlow_max32768.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 65536 --max-length 65536 --out %R%\ruler\Qwen3.5-4B_dkv_mid_streamlow_max65536.jsonl

rem 4. BASELINE needed either way: dense at 64k, now that the allocator fix lets
rem    it run there (it passes 65,536 on the ladder). Not a DKV setting, so the
rem    DKV_* variables above do not touch it.
set DKV_STREAMING_COMPRESS=
set DKV_PREFILL_LOWMEM=
set DKV_REMAT_CACHE=
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dense --min-length 65536 --max-length 65536 --baseline-params "{\"defrag\": true}" --out %R%\ruler\Qwen3.5-4B_dense_defrag_max65536.jsonl

echo STREAMLOW CAMPAIGN COMPLETE
