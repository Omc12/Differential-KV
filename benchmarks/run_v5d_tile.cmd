@echo off
rem V5-D -- granite reach of the codes store with a smaller prefill tile. V5-B's
rem codes4 ladder spilled at 98,304 (14.38 / 18.06 GB) where tiled KIVI-4 fits
rem (10.66 / 11.50) with 0.6 GB fewer stored bytes: the fused history tile
rem dequantizes 64 blocks of key and value codes in fp32 per layer call (~1.1 GB
rem beyond the low-rank tile). 16 blocks per tile bounds it 4x lower; tiles merge
rem by log-sum-exp, so the arithmetic is unchanged. ~45 min.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set DKV_STORE=codes
set DKV_QUANT_BITS=4
set DKV_PREFILL_SDPA=1
set DKV_PREFILL_SDPA_TILE=16
set DKV_STREAM_ELASTIC=1
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 98304 114688 --timeout 5400 --out paper\results\ladder\V5_granite_codes4_tile16_auto.jsonl
echo V5D COMPLETE
