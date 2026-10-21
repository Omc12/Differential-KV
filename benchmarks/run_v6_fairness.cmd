@echo off
rem V6 -- the two controls blind review asked for most.
rem   p1  dense_tiled reach: a 16-bit cache held in chunks and read by the same
rem       tiled, LSE-merged attention as DKV and tiled KIVI -- the dense
rem       ceiling with no history-sized transient. Ladders on both models.
rem   p2  DKV with no question information in residual selection
rem       (DKV_RESIDUAL_NO_QUERY=1): RULER at 131k on Qwen3.5-4B, the
rem       V3_BEYOND items, paired with DKV as shipped and the eviction arms.
rem   p3  question-aware chunk-wise eviction at 1.5x DKV's bytes
rem       (snapkv_chunked_q), same items.
rem RULER on the paper's prompt (no answer prefix). Resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=0
set NOP={}

if not "%1"=="" goto %1

:p1
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense_tiled --gen 128 --contexts 16384 20480 24576 28672 32768 36864 40960 --baseline-params "%NOP%" --timeout 7200 --out %R%\ladder\V6_granite_dense_tiled.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense_tiled --gen 128 --contexts 98304 122880 131072 163840 196608 229376 262144 --baseline-params "%NOP%" --timeout 7200 --out %R%\ladder\V6_Qwen3.5-4B_dense_tiled.jsonl
echo V6 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
setlocal
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set DKV_RESIDUAL_NO_QUERY=1
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\V6_Qwen3.5-4B_dkv_mid_noquery_131k.jsonl
endlocal
echo V6 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm snapkv_chunked_q --min-length 131072 --max-length 131072 --per-task 5 --baseline-params "{\"budget\": 23808, \"recency_window\": 2048, \"question_tail\": 64}" --out %R%\ruler\V6_Qwen3.5-4B_snapkv_chunked_q_eqbytes_131k.jsonl
echo V6 P3 COMPLETE
