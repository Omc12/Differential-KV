@echo off
rem V13m -- Phi-4-mini lowrank: U int8 one-scale-per-block (colscale) and joint K/V factorization (kvsplit).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true baseline colscale kvsplit
echo V13M COMPLETE
