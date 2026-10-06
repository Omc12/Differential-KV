@echo off
rem DKV_EXACT_LEAN=1: exact prefill attends raw history a tile at a time (KV-head
rem width, GQA folded, no whole-history copies) + the decode-cache memory gate.
rem Run when the GPU is free. ~1.5 h.
rem 0) unit tests  1) tier 2 vs the default exact path (should be equal up to
rem rounding)  2) exact-mode ceilings, granite and Qwen3.5-4B, vs
rem granite-4.2-8b_mid_nf4 / Qwen3.5-4B_mid_nf4 (exact: 16k / 98k) and the
rem preallocated dense cache (20k / 122,880).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
pushd ACTIVE_RUNTIME
"%PY%" -m pytest -q tests\test_exact_lean.py tests\test_prefill_sdpa.py tests\test_key_quant_hybrid.py
if errorlevel 1 (echo UNIT TESTS FAILED - stopping & popd & exit /b 1)
popd
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "exact_lean:DKV_EXACT_LEAN=1"
set DKV_EXACT_LEAN=1
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 16384 20480 24576 28672 --timeout 7200 --out paper\results\diag\TEST_granite_exact_lean_ladder.jsonl
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --gen 128 --contexts 98304 114688 122880 131072 --timeout 7200 --out paper\results\diag\TEST_Qwen3.5-4B_exact_lean_ladder.jsonl
echo EXACT LEAN COMPLETE
