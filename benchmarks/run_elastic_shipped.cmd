@echo off
rem D2 on the SHIPPED store: tier 2 vs plain streaming, and reach at 131k
rem (streaming auto at 16k, as the shipped ceiling ladder).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "stream:DKV_STREAMING_COMPRESS=1" "elastic_base_stream:DKV_STREAMING_COMPRESS=1,DKV_STREAM_ELASTIC=1"
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
set DKV_STREAM_ELASTIC=1
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 131072 --out paper\results\diag\TEST_granite_elastic_shipped_131k.jsonl
echo ELASTIC SHIPPED COMPLETE
