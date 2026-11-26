@echo off
rem V18 -- RULER at 32k: tiled KIVI-4 against DKV on its default store (codes,
rem 2026-11-26), on the V14 items (SmolLM3 all 13 tasks, the rest the 4-task
rem subset). Dense is V14's, paired on shared items; low-rank DKV is V15's
rem (pinned DKV_STORE=lowrank). Models as arguments run only those.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set DKV_STORE=codes
rem Generated RULER data (untracked); defaults to this checkout's copy.
if "%DATA%"=="" set DATA=%CD%\paper\results\ruler_data
set ALL13=niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
set SUB4=niah_multikey_2 niah_multivalue vt qa_1
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-7B-Instruct microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it google/gemma-4-12B-it mistralai/Ministral-3-8B-Instruct-2512-BF16

for %%M in (%MODELS%) do (
  echo === V18 %%M ===
  set T=%SUB4%
  if "%%M"=="HuggingFaceTB/SmolLM3-3B" set T=%ALL13%
  call :run %%M
  echo === V18 %%M DONE ===
)
echo V18 COMPLETE
goto :eof

:run
for %%A in (kivi4_tiled dkv) do (
  "%PY%" benchmarks\run_ruler_cuda.py --model %1 --arm %%A --preset mid --min-length 32768 --max-length 32768 --tasks %T% --per-task 1 --data "%DATA%\%~nx1" --out %R%\V18_%~nx1_%%A_32k.jsonl
)
goto :eof
