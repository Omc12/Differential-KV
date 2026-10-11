@echo off
rem Pre-flight for V3 part C2: hybrid streaming on Qwen3.5-4B (exact hybrid already
rem smoke-tested, KL 0.0006). Reuses finished arms.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
"%PY%" benchmarks\decode_fidelity.py --model Qwen/Qwen3.5-4B --gen 16 --lb --ruler 16384/niah_multikey_2:0-0 16384/niah_multikey_3:0-0 16384/qa_1:0-0 16384/qa_2:0-0 --arms base "hybrid:DKV_STORE=hybrid" "hybrid_stream:DKV_STORE=hybrid,DKV_STREAMING_COMPRESS=1" "stream:DKV_STREAMING_COMPRESS=1"
echo QWEN SMOKE COMPLETE
