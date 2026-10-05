@echo off
rem V3 PART B -- KIVI-4 with tiled attention (kivi4_tiled): same quantizer and
rem bytes as kivi4_chunked, but the 4-bit history is read a tile at a time, as
rem KIVI's own kernels and DKV's streaming path do. kivi4_chunked dequantized the
rem whole history every step, so its reach (49k granite, 98k Qwen) was a harness
rem limit. Every KIVI-4 row in the paper (reach and quality) comes from here.
rem Resumable. Est. ~6-7 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B

rem -- B1. ceilings (the ladder stops an arm after its first spill)
"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi4_tiled --gen 128 --contexts 16384 32768 49152 65536 98304 131072 --timeout 7200 --out %R%\ladder\V3_granite_kivi4_tiled.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms kivi4_tiled --gen 128 --contexts 65536 98304 131072 163840 196608 262144 --timeout 7200 --out %R%\ladder\V3_Qwen3.5-4B_kivi4_tiled.jsonl

rem -- B2. quality, same items as the DKV rows they are compared with
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm kivi4_tiled --num-samples 20 --max-length 12000 --out %R%\longbench\V3_granite-4.2-8b_kivi4_tiled_len12000.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_tiled --min-length 24576 --max-length 32768 --per-task 10 --out %R%\ruler\V3_BEYOND_granite_kivi4_tiled_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_tiled --min-length 32768 --max-length 32768 --per-task 10 --out %R%\ruler\V3_Qwen3.5-4B_kivi4_tiled_32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_tiled --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\V3_BEYOND_Qwen3.5-4B_kivi4_tiled_131k.jsonl

echo V3 PART B COMPLETE
