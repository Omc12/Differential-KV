@echo off
rem V16 -- is DKV on the codes store (the default since 2026-11-26) better than
rem KIVI-4 on quality, and faster? decode_fidelity.py on RULER 32k prompts:
rem dense's own answer forced step by step through each arm, so every step is a
rem paired comparison. Arms: KIVI-4 (tiled where the model allows, else
rem chunked; the row records which), DKV codes, DKV low-rank. Reports per-step
rem KL vs dense, top-1, dNLL, stored GB and decode tok/s (graph-free on both).
rem Models as arguments run only those.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-7B-Instruct microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it mistralai/Ministral-3-8B-Instruct-2512-BF16 ibm-granite/granite-4.2-8b

for %%M in (%MODELS%) do (
  echo === V16 %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 codes:DKV_STORE=codes lowrank:DKV_STORE=lowrank
  echo === V16 %%M DONE ===
)
echo V16 COMPLETE
