@echo off
rem CHUNK-WISE KIVI-4: the streaming alternative that keeps every token.
rem 0. gate: converted layers, real 4-bit bytes, near-dense distribution.
rem 1. ceiling, 128-token answers, both models.
rem 2. quality beyond the dense limit (same items as streaming DKV) and on
rem    LongBench at 12k (same items as every other arm).
rem Resumable: rerun after a power cut.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1

"%PY%" benchmarks\selfcheck_kivi_chunked.py
if errorlevel 1 (
  echo KIVI SELFCHECK FAILED -- stopping
  exit /b 1
)

"%PY%" benchmarks\context_ladder.py --model %QW% --arms kivi4_chunked --gen 128 --contexts 65536 98304 131072 163840 --out %R%\ladder\GEN128_Qwen3.5-4B_kivi4_chunked.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi4_chunked --gen 128 --contexts 16384 24576 32768 49152 --out %R%\ladder\GEN128_granite_kivi4_chunked.jsonl

"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_chunked --min-length 24576 --max-length 32768 --per-task 10 --out %R%\ruler\BEYOND_granite_kivi4_chunked_24k-32k.jsonl
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm kivi4_chunked --num-samples 20 --max-length 12000 --out %R%\longbench\granite-4.2-8b_kivi4_chunked_len12000.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_chunked --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\BEYOND_Qwen3.5-4B_kivi4_chunked_131k.jsonl

echo KIVI CAMPAIGN COMPLETE
