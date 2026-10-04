@echo off
rem D2: elastic streaming window (DKV_STREAM_ELASTIC=1) on the hybrid store.
rem v2: the gate now covers both mid-prefill compress paths (v1 gated only the
rem per-layer one and measured identical to plain streaming).
rem 1) quality on the 24 tier-2 prompts; 2) 65k granite point with and without
rem    elastic, to attribute the 65k spill seen with v1.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
set H=DKV_KEY_QUANT=pc4,DKV_RESID_ATTN=1,DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "fast_att:DKV_KEY_QUANT=pc4,DKV_RESID_ATTN=1" "fast_att_stream:%H%" "elastic_stream:%H%,DKV_STREAM_ELASTIC=1"
set DKV_KEY_QUANT=pc4
set DKV_RESID_ATTN=1
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 --out paper\results\diag\TEST_granite_hybrid_stream_65k.jsonl
set DKV_STREAM_ELASTIC=1
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 --out paper\results\diag\TEST_granite_hybrid_elastic2_65k.jsonl
echo D2 COMPLETE
