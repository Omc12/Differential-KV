@echo off
rem V22 -- decode_fidelity of the routed default (codes store, heads vote,
rem adaptive growth at any block count; V21) on the README's other four
rem models, whose rows predated the value fix and both router changes.
rem Same prompts as V16; kivi4 is cached. codes_v22 = vote (the V21 default,
rem since reverted); codes_v22_maxgrow = the default after V22b.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
if "%DATA%"=="" set DATA=%CD%\paper\results\ruler_data
if "%DKV_RULER_DATA%"=="" set DKV_RULER_DATA=%DATA%
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B microsoft/Phi-4-mini-reasoning Qwen/Qwen3.5-2B mistralai/Ministral-3-8B-Instruct-2512-BF16

for %%M in (%MODELS%) do (
  echo === V22 fidelity %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 codes_v22:DKV_STORE=codes,DKV_ROUTE_HEADS=vote codes_v22_maxgrow:DKV_STORE=codes,DKV_ROUTE_HEADS=max,DKV_ROUTE_ADAPTIVE_MIN_BLOCKS=0
  echo === V22 fidelity %%M DONE ===
)
echo V22 COMPLETE
