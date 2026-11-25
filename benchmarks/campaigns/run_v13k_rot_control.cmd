@echo off
rem V13k -- control for V13j: rotated pool on Qwen2.5-7B (passes under mid).
rem UNROTATED and rotates them at read; codes store (no read-time rotation) is
rem clean at 0.0017 while even exact-history decode (dense_arm) is 0.047.
rem rot1 / rot1_dense store post-RoPE keys, so nothing is rotated at read.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
"%PY%" colab\logit_fidelity.py --model Qwen/Qwen2.5-7B-Instruct --quant nf4 --ctx 32768 --arms dense_true baseline rot1
echo V13K COMPLETE
