@echo off
rem V5 -- KIVI's codes as a DKV store (DKV_STORE=codes: keys per-channel, values
rem per-token, 4 or 2 bits, group 32, the tiled KIVI baseline's layout), run by
rem DKV's own scheduler, router and read path. Each part pairs item for item with
rem a store already in paper\results: the tiled KIVI rows (same codes, no router,
rem no streaming scheduler) and DKV's default store. RULER on the paper's prompt
rem (no answer prefix), as every other RULER store. Resumable; parts in priority
rem order, each stands alone:
rem   p1  Qwen3.5-4B RULER at 131k, codes4           (~2 h)
rem   p2  granite RULER at 24k-32k, codes4           (~4 h)
rem   p3  reach: granite 98k-131k codes4/codes2, Qwen 131k-262k codes4  (~1.5 h)
rem   p4  TTFT and decode rate at the same rungs     (~1.5 h)
rem   p5  router at 131k: codes4 attending every block (~2.5 h)
rem   p6  granite RULER at 114k, codes2              (~4 h)
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
rem Offline: the hub re-sharded granite upstream; every store here used the cached revision.
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set DKV_STORE=codes
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set RULER_ANSWER_PREFIX=0
set G=64

if not "%1"=="" goto %1

:p1
setlocal
set DKV_QUANT_BITS=4
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\V5_Qwen3.5-4B_codes4_auto_131k.jsonl
endlocal
echo V5 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
setlocal
set DKV_QUANT_BITS=4
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 24576 --max-length 32768 --per-task 10 --out %R%\ruler\V5_granite_codes4_auto_24k-32k.jsonl
endlocal
echo V5 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
setlocal
set DKV_STREAM_AUTO_TOKENS=16384
set DKV_QUANT_BITS=4
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 98304 114688 131072 --timeout 7200 --out %R%\ladder\V5_granite_codes4_auto.jsonl
set DKV_QUANT_BITS=2
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 98304 114688 131072 --timeout 7200 --out %R%\ladder\V5_granite_codes2_auto.jsonl
set DKV_QUANT_BITS=4
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 131072 196608 262144 --timeout 7200 --out %R%\ladder\V5_Qwen3.5-4B_codes4_auto.jsonl
endlocal
echo V5 P3 COMPLETE
if not "%1"=="" goto :eof

:p4
setlocal
set DKV_QUANT_BITS=4
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dkv --gen %G% --contexts 32768 65536 98304 --out %R%\systems\V5_granite_codes4_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dkv --gen %G% --contexts 32768 131072 262144 --out %R%\systems\V5_Qwen3.5-4B_codes4_auto.jsonl
endlocal
echo V5 P4 COMPLETE
if not "%1"=="" goto :eof

:p5
setlocal
set DKV_QUANT_BITS=4
set DKV_STREAM_AUTO_TOKENS=98304
set DKV_TOPK_BLOCKS=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\V5_Qwen3.5-4B_codes4_attendall_131k.jsonl
endlocal
echo V5 P5 COMPLETE
if not "%1"=="" goto :eof

:p6
setlocal
set DKV_QUANT_BITS=2
set DKV_STREAM_AUTO_TOKENS=24576
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 114688 --max-length 114688 --per-task 3 --out %R%\ruler\V5_granite_codes2_114k.jsonl
endlocal
echo V5 P6 COMPLETE
