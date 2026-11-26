@echo off
rem V19 -- V18's DKV arm again after the value codes stopped being anchor-
rem relative (2026-11-26): values now reach the codec as they are, as in KIVI.
rem Same items, same flags; V18's KIVI-4 arms are unaffected and are reused.
rem Models as arguments run only those.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set DKV_STORE=codes
if "%DATA%"=="" set DATA=%CD%\paper\results\ruler_data
set ALL13=niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
set SUB4=niah_multikey_2 niah_multivalue vt qa_1
set MODELS=%*
if "%MODELS%"=="" set MODELS=Qwen/Qwen2.5-7B-Instruct HuggingFaceTB/SmolLM3-3B microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it google/gemma-4-12B-it mistralai/Ministral-3-8B-Instruct-2512-BF16

for %%M in (%MODELS%) do (
  echo === V19 %%M ===
  set T=%SUB4%
  if "%%M"=="HuggingFaceTB/SmolLM3-3B" set T=%ALL13%
  call :run %%M
  echo === V19 %%M DONE ===
)
echo V19 COMPLETE
goto :eof

:run
"%PY%" benchmarks\run_ruler_cuda.py --model %1 --arm dkv --preset mid --min-length 32768 --max-length 32768 --tasks %T% --per-task 1 --data "%DATA%\%~nx1" --out %R%\V19_%~nx1_dkv_32k.jsonl
goto :eof
