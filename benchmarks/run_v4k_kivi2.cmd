@echo off
rem V4-K -- tiled KIVI-2 (the published configuration) where free-form output
rem shows: LongBench at 12k, the same 120 items as every LongBench row. RULER
rem beyond dense's limit for KIVI-2 was dropped for time; the paper states
rem its quality there as unmeasured. Resumable.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set DKV_TRITON_STRICT=1

"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm kivi2_tiled --num-samples 20 --max-length 12000 --out %R%\longbench\V4_granite-4.2-8b_kivi2_tiled_len12000.jsonl

echo V4-K COMPLETE
