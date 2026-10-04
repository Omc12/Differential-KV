@echo off
rem E3: tiled KIVI-4 (kivi4_tiled) -- same quantizer and bytes as kivi4_chunked,
rem attention reads the 4-bit history a tile at a time. Small tests only:
rem 1) reach smoke: granite 65k (chunked spilled) and 98k;
rem 2) agreement: RULER 24k, 1 item per task, tiled vs chunked on the same items.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set GR=ibm-granite/granite-4.2-8b
"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi4_tiled --gen 128 --contexts 65536 98304 --out paper\results\diag\TEST_granite_kivi4_tiled_ladder.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_tiled --min-length 24576 --max-length 24576 --per-task 1 --out paper\results\diag\TEST_granite_kivi4_tiled_ruler24k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_chunked --min-length 24576 --max-length 24576 --per-task 1 --out paper\results\diag\TEST_granite_kivi4_chunked_ruler24k.jsonl
echo E3 COMPLETE
