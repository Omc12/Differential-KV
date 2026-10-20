@echo off
rem V4 PART G -- LongBench (granite, 12k, 20 per task) for tiled KIVI-2, on the
rem same items as every other LongBench row. LongBench has no answer-prefix
rem field, so the RULER harness fix does not touch these stores.
rem Resumable. Est. ~1.5 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b

"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm kivi2_tiled --num-samples 20 --max-length 12000 --out %R%\longbench\V4_granite-4.2-8b_kivi2_tiled_len12000.jsonl

echo V4 PART G COMPLETE
