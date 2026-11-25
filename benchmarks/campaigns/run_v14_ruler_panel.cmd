@echo off
rem V14 -- short RULER pass on the V13 panel: dense vs DKV (shipped mid preset,
rem nothing set per model) at 32k, 13 tasks x 1 item (2 generated per task).
rem A smoke test for catastrophic failure across architectures, not a ranking:
rem n=13 per arm (TASKS narrows it; the V14 subset run used 4). Models as
rem arguments run only those.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
if "%TASKS%"=="" set TASKS=niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B google/gemma-4-e2b-it google/gemma-4-e4b-it google/gemma-4-12B-it mistralai/Ministral-3-8B-Instruct-2512-BF16 ibm-granite/granite-4.2-8b

for %%M in (%MODELS%) do (
  echo === V14 %%M ===
  for %%A in (dense dkv) do (
    "%PY%" benchmarks\run_ruler_cuda.py --model %%M --arm %%A --preset mid --min-length 32768 --max-length 32768 --tasks %TASKS% --per-task 1 --out %R%\V14_%%~nxM_%%A_32k.jsonl
  )
  echo === V14 %%M DONE ===
)
echo V14 COMPLETE
