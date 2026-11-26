@echo off
rem V15 -- the V14 RULER smoke test's DKV arm rerun after the residual
rem run-reservation fix (DKV_RESIDUAL_RUN_SHARE, 2026-11-26). Dense does not
rem read residual selection, so V14's dense arms are paired against unchanged.
rem Same items as V14: SmolLM3 all 13 tasks, the rest the 4-task subset.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
rem Pinned: codes became the default store the same day; this run measures
rem the low-rank store with the residual fix.
set DKV_STORE=lowrank
set ALL13=niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
set SUB4=niah_multikey_2 niah_multivalue vt qa_1

echo === V15 HuggingFaceTB/SmolLM3-3B ===
"%PY%" benchmarks\run_ruler_cuda.py --model HuggingFaceTB/SmolLM3-3B --arm dkv --preset mid --min-length 32768 --max-length 32768 --tasks %ALL13% --per-task 1 --out %R%\V15_SmolLM3-3B_dkv_32k.jsonl
for %%M in (Qwen/Qwen2.5-7B-Instruct microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it google/gemma-4-12B-it mistralai/Ministral-3-8B-Instruct-2512-BF16) do (
  echo === V15 %%M ===
  "%PY%" benchmarks\run_ruler_cuda.py --model %%M --arm dkv --preset mid --min-length 32768 --max-length 32768 --tasks %SUB4% --per-task 1 --out %R%\V15_%%~nxM_dkv_32k.jsonl
)
echo V15 RULER COMPLETE
