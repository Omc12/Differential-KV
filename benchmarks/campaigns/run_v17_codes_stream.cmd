@echo off
rem V17 -- the codes store built the way KIVI builds its cache (DKV_CODES_STREAM=1:
rem each block quantized as soon as it is complete, every prefill chunk attends
rem all 4-bit history, no exact-history elastic window), against two KIVI-4
rem references: KIVI as published (128-token 16-bit window, tiled) and KIVI given
rem the SAME 16-bit window DKV keeps at decode (KIVI_RESIDUAL=dkv), so a gap
rem cannot come from window size. codes_factor = the codes store with the old,
rem never-read value factorization (DKV_CODES_FACTOR=1), the A/B for skipping it.
rem Same prompts and output folder as V16, whose dense controls are reused; any
rem arm already in the folder is reused too (delete its json to rerun it).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-7B-Instruct microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it mistralai/Ministral-3-8B-Instruct-2512-BF16 ibm-granite/granite-4.2-8b

for %%M in (%MODELS%) do (
  echo === V17 %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 kivi4w:KIVI_RESIDUAL=dkv codes:DKV_STORE=codes codes_factor:DKV_STORE=codes,DKV_CODES_FACTOR=1 codes_stream:DKV_STORE=codes,DKV_CODES_STREAM=1 lowrank:DKV_STORE=lowrank
  echo === V17 %%M DONE ===
)
echo V17 COMPLETE
