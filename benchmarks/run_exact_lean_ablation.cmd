@echo off
rem Which half of DKV_EXACT_LEAN gives the reach, which costs decode speed:
rem tiled exact prefill alone (decode-cache gate forced off, DKV_REMAT_GATE=0).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "exact_lean:DKV_EXACT_LEAN=1" "lean_nogate:DKV_EXACT_LEAN=1,DKV_REMAT_GATE=0"
set DKV_EXACT_LEAN=1
set DKV_REMAT_GATE=0
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 20480 24576 --timeout 7200 --out paper\results\diag\TEST_granite_exact_lean_nogate.jsonl
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --gen 128 --contexts 122880 131072 --timeout 7200 --out paper\results\diag\TEST_Qwen3.5-4B_exact_lean_nogate.jsonl
echo ABLATION COMPLETE
