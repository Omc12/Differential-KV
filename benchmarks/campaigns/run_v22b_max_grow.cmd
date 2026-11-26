@echo off
rem V22b -- SmolLM3 regressed under DKV_ROUTE_HEADS=vote (decode KL 0.044 vs
rem 0.006 with max; adaptive growth neutral). Measure the max rule WITH growth
rem at any block count on the two models V21 was decided on: Qwen2.5-7B
rem (fidelity + RULER 4 items) and granite (fidelity).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
if "%DATA%"=="" set DATA=%CD%\paper\results\ruler_data
if "%DKV_RULER_DATA%"=="" set DKV_RULER_DATA=%DATA%

echo === V22b RULER Qwen2.5-7B max + growth ===
set DKV_ROUTE_HEADS=max
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen2.5-7B-Instruct --arm dkv --preset mid --min-length 32768 --max-length 32768 --tasks niah_multikey_2 niah_multivalue vt qa_1 --per-task 1 --data "%DATA%\Qwen2.5-7B-Instruct" --out paper\results\ruler\V22b_Qwen2.5-7B-Instruct_dkv_maxgrow_32k.jsonl
set DKV_ROUTE_HEADS=
echo === V22b RULER DONE ===

for %%M in (Qwen/Qwen2.5-7B-Instruct ibm-granite/granite-4.2-8b) do (
  echo === V22b fidelity %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 codes_v22_maxgrow:DKV_STORE=codes,DKV_ROUTE_HEADS=max,DKV_ROUTE_ADAPTIVE_MIN_BLOCKS=0
  echo === V22b fidelity %%M DONE ===
)
echo V22b COMPLETE
