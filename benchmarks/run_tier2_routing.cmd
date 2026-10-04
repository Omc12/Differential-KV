@echo off
rem Tier 2, batch C: routing at long context on Qwen3.5-4B (exact prefill),
rem 12 RULER prompts at 32k and 64k, against dense. Existing switches only.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
"%PY%" benchmarks\decode_fidelity.py --model Qwen/Qwen3.5-4B --gen 24 --lb ^
  --ruler 32768/niah_multikey_2:0-1 32768/niah_multiquery:0-0 32768/qa_1:0-0 32768/niah_multivalue:0-0 32768/niah_single_3:0-0 65536/niah_multikey_2:0-1 65536/niah_multiquery:0-0 65536/qa_1:0-0 65536/niah_multivalue:0-0 65536/niah_single_3:0-0 ^
  --arms base "all:DKV_TOPK_BLOCKS=0,DKV_BLOCKS_PER_CHUNK=256" "top32:DKV_TOPK_BLOCKS=32" "frac50:DKV_TOPK_FRAC=0.5"
echo ROUTING COMPLETE
