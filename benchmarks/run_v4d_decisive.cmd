@echo off
rem V4-D -- the one length where DKV's reach claim is decided: granite at 114,688
rem tokens, past where tiled KIVI-4 fits (98k clean, 131k spilled) and where DKV
rem and tiled KIVI-2 both still fit. Part 1 confirms reach at that rung for all
rem three; part 2 is RULER there, DKV (as shipped: auto, N=24,576, so it streams)
rem against tiled KIVI-2, 3 items per task, on the paper's RULER prompt (no answer prefix,
rem as every other RULER store, so it pairs and plots with them). Resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set DKV_TRITON_STRICT=1

if "%1"=="ruler" goto ruler

"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi4_tiled --gen 128 --contexts 114688 --timeout 3600 --out %R%\ladder\V4D_granite_kivi4_tiled_114k.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi2_tiled --gen 128 --contexts 114688 --timeout 3600 --out %R%\ladder\V4D_granite_kivi2_tiled_114k.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=24576
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 114688 --timeout 3600 --out %R%\ladder\V4D_granite_dkv_auto_114k.jsonl
endlocal
echo V4-D REACH COMPLETE
goto :eof

:ruler
set RULER_ANSWER_PREFIX=0
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=24576
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 114688 --max-length 114688 --per-task 3 --out %R%\ruler\V4D_granite-4.2-8b_dkv_mid_114k.jsonl
endlocal
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi2_tiled --min-length 114688 --max-length 114688 --per-task 3 --out %R%\ruler\V4D_granite-4.2-8b_kivi2_tiled_114k.jsonl
echo V4-D RULER COMPLETE
