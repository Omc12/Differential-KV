@echo off
rem MERGE CHECK -- the streaming fixes are now default-on; prove the exact path
rem did not change. Same RULER items with the three fixes forced OFF and with
rem the new defaults; the generated text must match item for item. Then one
rem ladder rung each way, so any memory change on the exact path is measured,
rem not assumed. Compare with:  python benchmarks\compare_merge_check.py
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\merge_check
set DKV_TRITON_STRICT=1
if not exist %R% mkdir %R%

set DKV_PREFILL_LOWMEM=0
set DKV_ROUTER_SLOT_DEQUANT=0
set DKV_CLAMP_DECODE_RANK=0
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 16384 --max-length 16384 --per-task 2 --out %R%\qwen16k_off.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset mid --min-length 8192 --max-length 8192 --per-task 2 --out %R%\granite8k_off.jsonl
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --contexts 16384 --out %R%\granite_ladder16k_off.jsonl
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --contexts 65536 --out %R%\qwen_ladder64k_off.jsonl
set DKV_PREFILL_LOWMEM=
set DKV_ROUTER_SLOT_DEQUANT=
set DKV_CLAMP_DECODE_RANK=

"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen3.5-4B --arm dkv --preset mid --min-length 16384 --max-length 16384 --per-task 2 --out %R%\qwen16k_on.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model ibm-granite/granite-4.2-8b --arm dkv --preset mid --min-length 8192 --max-length 8192 --per-task 2 --out %R%\granite8k_on.jsonl
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --contexts 16384 --out %R%\granite_ladder16k_on.jsonl
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --contexts 65536 --out %R%\qwen_ladder64k_on.jsonl

"%PY%" benchmarks\compare_merge_check.py
echo MERGE CHECK COMPLETE
