@echo off
rem V8 -- fill granite-4.2-8b's RULER curve in Figure 1 between 32k and 114k,
rem and give its eviction arms the lengths dense was scored at.
rem   p0  generate RULER items at 65,536 tokens (5 per task)
rem   p1  DKV/mid at 65k, as V3_BEYOND (lowrank, elastic, auto 16384)
rem   p2  tiled KIVI-4 at 65k (its ladder runs clean to 98k)
rem   p3  chunk-wise H2O and StreamingLLM at 4k-16k (10 per task)
rem   p4  SnapKV at 4k-16k (10 per task), the lengths it fits
rem RULER on the paper's prompt (no answer prefix). Every phase is resumable:
rem re-running a phase skips the items already in its store.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=0
set H2={\"budget\": 2048, \"recency_window\": 512, \"prefill_chunk\": 256}
set SL={\"n_sink\": 4, \"recency_window\": 2044}
set SN={\"budget\": 2016, \"window\": 32}

if not "%1"=="" goto %1

:p0
setlocal
set MODEL=%GR%
set LENGTHS=65536
set NSAMP=5
"C:\Program Files\Git\bin\bash.exe" benchmarks/generate_ruler_data.sh
endlocal
echo V8 P0 COMPLETE
if not "%1"=="" goto :eof

:p1
setlocal
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 65536 --max-length 65536 --per-task 5 --out %R%\ruler\V8_granite_dkv_mid_auto_65k.jsonl
endlocal
echo V8 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_tiled --min-length 65536 --max-length 65536 --per-task 5 --baseline-params "{}" --out %R%\ruler\V8_granite_kivi4_tiled_65k.jsonl
echo V8 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm h2o_chunked --min-length 4096 --max-length 16384 --per-task 10 --baseline-params "%H2%" --out %R%\ruler\V8_granite_h2o_chunked_4k-16k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm streamingllm_chunked --min-length 4096 --max-length 16384 --per-task 10 --baseline-params "%SL%" --out %R%\ruler\V8_granite_sl_chunked_4k-16k.jsonl
echo V8 P3 COMPLETE
if not "%1"=="" goto :eof

:p4
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm snapkv --min-length 4096 --max-length 16384 --per-task 10 --baseline-params "%SN%" --out %R%\ruler\V8_granite_snapkv_4k-16k.jsonl
echo V8 P4 COMPLETE
