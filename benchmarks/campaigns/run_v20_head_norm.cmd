@echo off
rem V20 -- the block router normalises each head by its own softmax before the
rem cross-head max (DKV_ROUTE_HEAD_NORM, default on). Short check against V19
rem (routed, raw logits) and V19b (every block read):
rem (1) RULER 32k on Qwen2.5-7B, the V19 four-task subset, routed default.
rem (2) decode_fidelity on the V16 prompts for Qwen2.5-7B and granite, one new
rem     arm; V19b's codes_v19 / codes_all_v19 arms are the comparison.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
if "%DATA%"=="" set DATA=%CD%\paper\results\ruler_data
if "%DKV_RULER_DATA%"=="" set DKV_RULER_DATA=%DATA%
set SUB4=niah_multikey_2 niah_multivalue vt qa_1

echo === V20 RULER Qwen2.5-7B routed, head norm ===
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen2.5-7B-Instruct --arm dkv --preset mid --min-length 32768 --max-length 32768 --tasks %SUB4% --per-task 1 --data "%DATA%\Qwen2.5-7B-Instruct" --out paper\results\ruler\V20_Qwen2.5-7B-Instruct_dkv_32k.jsonl
echo === V20 RULER DONE ===

for %%M in (Qwen/Qwen2.5-7B-Instruct ibm-granite/granite-4.2-8b) do (
  echo === V20 fidelity %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 codes_v19:DKV_STORE=codes codes_all_v19:DKV_STORE=codes,DKV_TOPK_BLOCKS=0,DKV_BLOCKS_PER_CHUNK=256 codes_v20:DKV_STORE=codes
  echo === V20 fidelity %%M DONE ===
)
echo V20 COMPLETE
