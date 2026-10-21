@echo off
rem V5-B -- the rest of V5, cut to what decides a claim (after p1, Qwen 131k RULER).
rem   reach  where tiled KIVI-4 stops: granite codes4 at 98k/114k/131k (tiled KIVI-4:
rem          98k clean, 114k spilled) and Qwen codes4 at 262k. One call per rung
rem          also gives TTFT and decode rate, comparable with the tiled KIVI ladders.
rem   ruler  granite 32k, 5 per task, codes4: items 0-4 of the tiled KIVI-4 and
rem          DKV/mid stores at 32k, so it pairs with both.
rem Resumable. Est. ~2.5 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set DKV_STORE=codes
set DKV_QUANT_BITS=4
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set RULER_ANSWER_PREFIX=0

setlocal
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 98304 114688 131072 --timeout 7200 --out %R%\ladder\V5_granite_codes4_auto.jsonl
endlocal
setlocal
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 131072 262144 --timeout 7200 --out %R%\ladder\V5_Qwen3.5-4B_codes4_auto.jsonl
endlocal
echo V5B REACH COMPLETE
setlocal
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 32768 --max-length 32768 --per-task 5 --out %R%\ruler\V5_granite_codes4_auto_32k.jsonl
endlocal
echo V5B COMPLETE
