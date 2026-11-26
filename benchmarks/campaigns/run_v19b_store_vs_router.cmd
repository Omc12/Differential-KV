@echo off
rem V19b -- after the value fix (values coded as they are): is what is left of
rem the gap to KIVI-4 on Qwen2.5-7B / granite the STORE or the ROUTER?
rem (1) RULER 32k on Qwen2.5-7B with every compressed block read
rem     (DKV_TOPK_BLOCKS=0), against V19's routed default and V18's KIVI-4.
rem (2) decode_fidelity on the V16 prompts, new arm names so V17b's pre-fix
rem     arms stay as they were: routed default and every block read.
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

echo === V19b RULER Qwen2.5-7B all blocks ===
set DKV_STORE=codes
set DKV_TOPK_BLOCKS=0
set DKV_BLOCKS_PER_CHUNK=256
"%PY%" benchmarks\run_ruler_cuda.py --model Qwen/Qwen2.5-7B-Instruct --arm dkv --preset mid --min-length 32768 --max-length 32768 --tasks %SUB4% --per-task 1 --data "%DATA%\Qwen2.5-7B-Instruct" --out paper\results\ruler\V19b_Qwen2.5-7B-Instruct_dkv_all_32k.jsonl
set DKV_STORE=
set DKV_TOPK_BLOCKS=
set DKV_BLOCKS_PER_CHUNK=
echo === V19b RULER DONE ===

for %%M in (Qwen/Qwen2.5-7B-Instruct ibm-granite/granite-4.2-8b) do (
  echo === V19b fidelity %%M ===
  "%PY%" benchmarks\decode_fidelity.py --model %%M --gen 32 --lb --ruler 32768/niah_multikey_2:0-0 32768/niah_multivalue:0-0 32768/vt:0-0 32768/qa_1:0-0 --out paper\results\diag\v16 --arms kivi4 codes_v19:DKV_STORE=codes codes_all_v19:DKV_STORE=codes,DKV_TOPK_BLOCKS=0,DKV_BLOCKS_PER_CHUNK=256
  echo === V19b fidelity %%M DONE ===
)
echo V19b COMPLETE
