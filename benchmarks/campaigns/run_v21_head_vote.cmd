@echo off
rem V21 -- the block router combines heads by per-head softmax VOTES
rem (DKV_ROUTE_HEADS=vote, default) instead of the largest raw logit of any
rem head (max). V20's normalise-then-max made Qwen2.5-7B worse; see the log.
rem Also: the adaptive block count with its 2*K cutoff removed
rem (DKV_ROUTE_ADAPTIVE_MIN_BLOCKS=0), so it can grow at 32k.
rem decode_fidelity on the V16 prompts; kivi4 / codes_all_v19 are cached.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
if "%DATA%"=="" set DATA=%CD%\paper\results\ruler_data
if "%DKV_RULER_DATA%"=="" set DKV_RULER_DATA=%DATA%

for %%M in (Qwen/Qwen2.5-7B-Instruct ibm-granite/granite-4.2-8b) do (
  echo === V21 fidelity %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 codes_all_v19:DKV_STORE=codes,DKV_TOPK_BLOCKS=0,DKV_BLOCKS_PER_CHUNK=256 codes_v21_max:DKV_STORE=codes,DKV_ROUTE_HEADS=max codes_v21:DKV_STORE=codes codes_v21_grow:DKV_STORE=codes,DKV_ROUTE_ADAPTIVE_MIN_BLOCKS=0
  echo === V21 fidelity %%M DONE ===
)
echo V21 COMPLETE
