@echo off
rem V10 -- the past-the-limit comparisons again with RULER's answer prefix
rem attached (RULER_ANSWER_PREFIX=1, the harness default), at 10 items per task
rem instead of 5, so variable tracking and common-word extraction score as
rem RULER intends and every paired interval narrows. Items are the same
rem generated ones, the first 10 per task. Priority order; every phase is
rem resumable (re-running skips the items already in its store).
rem   p1  Qwen3.5-4B 131k  DKV/mid, streaming (as V3_BEYOND)
rem   p2  Qwen3.5-4B 131k  dense on DKV's tiled read path
rem   p3  granite 24k      DKV/mid, streaming (as V3_BEYOND)
rem   p4  granite 24k      dense on DKV's tiled read path
rem   p5  Qwen3.5-4B 131k  tiled KIVI-4
rem   p6  Qwen3.5-4B 131k  KIVI's 4-bit codes as a DKV store
rem   p7  Qwen3.5-4B 131k  chunk-wise SnapKV told the question (23,808 tokens)
rem   p8  Qwen3.5-4B 131k  chunk-wise StreamingLLM at 1.5x DKV's bytes
rem   p9  Qwen3.5-4B 131k  DKV/mid without the question look-ahead
rem   p10 granite 24k      tiled KIVI-4
rem   p11 Qwen3.5-4B 131k  chunk-wise StreamingLLM, 2,048 tokens
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set Q=--min-length 131072 --max-length 131072 --per-task 10
set G=--min-length 24576 --max-length 24576 --per-task 10

if "%1"=="resume" goto resume
if not "%1"=="" goto %1

:p1
setlocal
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q% --out %R%\ruler\V10P_Qwen3.5-4B_dkv_mid_auto_131k.jsonl
endlocal
echo V10 P1 COMPLETE
if not "%1"=="" goto :eof

:p2
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense_tiled %Q% --baseline-params "{}" --out %R%\ruler\V10P_Qwen3.5-4B_dense_tiled_131k.jsonl
echo V10 P2 COMPLETE
if not "%1"=="" goto :eof

:p3
setlocal
set DKV_STORE=lowrank
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid %G% --out %R%\ruler\V10P_granite_dkv_mid_auto_24k.jsonl
endlocal
echo V10 P3 COMPLETE
if not "%1"=="" goto :eof

:p4
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dense_tiled %G% --baseline-params "{}" --out %R%\ruler\V10P_granite_dense_tiled_24k.jsonl
echo V10 P4 COMPLETE
if not "%1"=="" goto :eof

:p5
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_tiled %Q% --baseline-params "{}" --out %R%\ruler\V10P_Qwen3.5-4B_kivi4_tiled_131k.jsonl
echo V10 P5 COMPLETE
if not "%1"=="" goto :eof

:p6
setlocal
set DKV_STORE=codes
set DKV_QUANT_BITS=4
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q% --out %R%\ruler\V10P_Qwen3.5-4B_codes4_auto_131k.jsonl
endlocal
echo V10 P6 COMPLETE
if not "%1"=="" goto :eof

:p7
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm snapkv_chunked_q %Q% --baseline-params "{\"budget\": 23808, \"recency_window\": 2048, \"question_tail\": 64}" --out %R%\ruler\V10P_Qwen3.5-4B_snapkv_chunked_q_eqbytes_131k.jsonl
echo V10 P7 COMPLETE
if not "%1"=="" goto :eof

:p8
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked %Q% --baseline-params "{\"n_sink\": 4, \"recency_window\": 23804}" --out %R%\ruler\V10P_Qwen3.5-4B_sl_chunked_eqbytes_131k.jsonl
echo V10 P8 COMPLETE
if not "%1"=="" goto :eof

:p9
setlocal
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
set DKV_RESIDUAL_NO_QUERY=1
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %Q% --out %R%\ruler\V10P_Qwen3.5-4B_dkv_mid_noquery_131k.jsonl
endlocal
echo V10 P9 COMPLETE
if not "%1"=="" goto :eof

:p10
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_tiled %G% --baseline-params "{}" --out %R%\ruler\V10P_granite_kivi4_tiled_24k.jsonl
echo V10 P10 COMPLETE
if not "%1"=="" goto :eof

:p11
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked %Q% --baseline-params "{\"n_sink\": 4, \"recency_window\": 2044}" --out %R%\ruler\V10P_Qwen3.5-4B_sl_chunked_131k.jsonl
echo V10 P11 COMPLETE
goto :eof

rem After a power cut: run every phase in order; each skips the items already
rem in its store, so finished phases cost only a model load.
:resume
for %%P in (p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11) do call "%~f0" %%P
echo V10 RESUME COMPLETE
