@echo off
rem Check of the incremental-prefill raw-history fix (tiled, was eager): Qwen3.5-4B
rem streaming reach (spilled at 131k at 18.58 GB; paper ladder 262k at 7.92 GB)
rem and fidelity on granite streaming vs the pre-fix elastic arm.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model Qwen/Qwen3.5-4B --arms dkv --gen 128 --contexts 131072 196608 262144 --timeout 7200 --out paper\results\diag\TEST_Qwen3.5-4B_inc_tiled_auto.jsonl
endlocal
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "stream:DKV_STREAMING_COMPRESS=1" "elastic_base_stream:DKV_STREAMING_COMPRESS=1,DKV_STREAM_ELASTIC=1" "inc_tiled_stream:DKV_STREAMING_COMPRESS=1,DKV_STREAM_ELASTIC=1,DKV_EXACT_LEAN=1"
echo CHECK COMPLETE
