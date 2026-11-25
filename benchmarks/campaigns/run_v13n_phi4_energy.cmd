@echo off
rem V13n -- Phi-4-mini: is it the energy truncation (joint path keeps the smallest k reaching svd_energy; split path keeps full rank)? energy1 = no truncation.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true energy1
echo V13N COMPLETE
