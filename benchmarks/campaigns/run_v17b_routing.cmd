@echo off
rem V17b -- after V17: (1) is the codes store's gap to KIVI-4 on some models the
rem ROUTER? codes_all attends every compressed block (DKV_TOPK_BLOCKS=0), the
rem same store and codes, only the routed set differs. (2) Phi-4-mini's KIVI arms
rem rerun after the kv_baselines sliding-window fix (delete their json first).
rem (3) codes_elastic: streaming prefill WITH the elastic window -- history
rem stays exact while the card-derived budget holds it, only the overflow is
rem compressed -- the candidate default once granite spilled at 32k (13.8 GB).
rem Every other arm in paper\results\diag\v16 is reused, not rerun.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B Qwen/Qwen2.5-7B-Instruct microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it mistralai/Ministral-3-8B-Instruct-2512-BF16 ibm-granite/granite-4.2-8b

for %%M in (%MODELS%) do (
  echo === V17b %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 kivi4w:KIVI_RESIDUAL=dkv codes:DKV_STORE=codes codes_all:DKV_STORE=codes,DKV_TOPK_BLOCKS=0,DKV_BLOCKS_PER_CHUNK=256 codes_factor:DKV_STORE=codes,DKV_CODES_FACTOR=1 codes_stream:DKV_STORE=codes,DKV_CODES_STREAM=1 codes_elastic:DKV_STORE=codes,DKV_STREAMING_COMPRESS=1 lowrank:DKV_STORE=lowrank
  echo === V17b %%M DONE ===
)
echo V17b COMPLETE
