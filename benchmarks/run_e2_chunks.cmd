@echo off
rem E2: larger prefill chunks (DKV_PREFILL_CHUNK_SIZE; rounded up to whole
rem blocks, so 2048 -> 2050 = 2 blocks/chunk, 4096 -> 4100 = 4). No code change.
rem 1) fidelity on the 24 tier-2 prompts, exact and streaming hybrid;
rem 2) prefill time and peak at 32k, streaming hybrid, chunk 1024/2048/4096.
rem E1 rides along: DKV_PREFILL_SDPA=1 (fused history attention), arm sdpa_stream
rem and a 32k ladder point; it only touches streaming (exact prefill reads raw KV).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set LB=qasper:13-13 qasper:15-16 multifieldqa_en:1-1 multifieldqa_en:3-3 multifieldqa_en:5-5 multifieldqa_en:9-9 hotpotqa:0-2 narrativeqa:0-0 narrativeqa:4-4
set X=DKV_KEY_QUANT=pc4,DKV_RESID_ATTN=1
set H=%X%,DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\decode_fidelity.py --gen 24 --lb %LB% ^
  --ruler 16384/niah_multikey_2:0-1 16384/niah_multikey_3:0-1 16384/qa_1:0-1 16384/qa_2:0-1 16384/niah_multivalue:0-0 16384/niah_multiquery:0-0 16384/niah_single_3:0-0 16384/fwe:0-0 ^
  --arms base "fast_att:%X%" "fast_att_stream:%H%" "chunk2k:%X%,DKV_PREFILL_CHUNK_SIZE=2048" "chunk4k:%X%,DKV_PREFILL_CHUNK_SIZE=4096" "chunk2k_stream:%H%,DKV_PREFILL_CHUNK_SIZE=2048" "chunk4k_stream:%H%,DKV_PREFILL_CHUNK_SIZE=4096" "sdpa_stream:%H%,DKV_PREFILL_SDPA=1"
set DKV_KEY_QUANT=pc4
set DKV_RESID_ATTN=1
set DKV_STREAMING_COMPRESS=1
for %%C in (1024 2048 4096) do (
  set DKV_PREFILL_CHUNK_SIZE=%%C
  "%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 32 --contexts 32768 --out paper\results\diag\TEST_granite_e2_chunk%%C_32k.jsonl
)
set DKV_PREFILL_CHUNK_SIZE=
set DKV_PREFILL_SDPA=1
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 32 --contexts 32768 --out paper\results\diag\TEST_granite_e1_sdpa_32k.jsonl
echo E2 COMPLETE
