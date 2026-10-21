@echo off
rem V5-E -- granite reach of the codes store under the fixed-window scheduler.
rem The codes store spilled on granite at 98,304 (64-block tile) and at 65,536
rem (16-block tile, V5-D) where tiled KIVI-4 fits, with peak minus stored bytes
rem ~8.5 GB against KIVI's ~5.9. The elastic scheduler holds exact history up
rem to a 3.5 GB headroom tuned on the low-rank payload, then encodes it all at
rem the prefill boundary; this run removes that variable. ~30 min.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
set DKV_STORE=codes
set DKV_QUANT_BITS=4
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=0
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 98304 --timeout 5400 --out paper\results\ladder\V5_granite_codes4_fixedwin_auto.jsonl
echo V5E COMPLETE
