@echo off
REM ── Run a command with the MSVC toolchain on PATH ────────────────────────────
REM
REM WHY THIS EXISTS
REM   torch.compile / Inductor needs a host C++ compiler. On Windows that is
REM   cl.exe, which MSVC only puts on PATH inside a "Developer Command Prompt" --
REM   installing Build Tools is NOT enough. Without it, DKV's decode path logs
REM
REM       [DKV JIT] _reconstruct_and_score: backend failed at call time
REM       (RuntimeError: Compiler: cl is not found.) Falling back to eager
REM
REM   and every latency/throughput number that follows UNDERSTATES DKV, because
REM   the Inductor-wrapped reconstruction runs unfused. The fallback is partial,
REM   not total: the Triton fused decode kernel still compiles and runs, so the
REM   symptom is a quiet slowdown rather than an error.
REM
REM   Build Tools 2022 is already installed on this box; only the environment was
REM   missing. Prefix EVERY timing run with this script.
REM
REM USAGE
REM   tools\with_msvc.bat python colab\validate_cuda_dkv.py
REM
REM VERIFY
REM   After running, confirm the log contains NO "did NOT compile" warning.

setlocal
set "_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "_VSPATH="
if exist "%_VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%_VSWHERE%" -latest -products * -property installationPath`) do set "_VSPATH=%%i"
)
if not defined _VSPATH (
    echo [with_msvc] ERROR: no Visual Studio / Build Tools installation found via vswhere.
    echo [with_msvc] Inductor will fall back to eager; timings would understate DKV.
    exit /b 1
)

set "_VCVARS=%_VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%_VCVARS%" (
    echo [with_msvc] ERROR: vcvars64.bat not found under "%_VSPATH%".
    exit /b 1
)

call "%_VCVARS%" >nul 2>&1
where cl >nul 2>&1
if errorlevel 1 (
    echo [with_msvc] ERROR: cl.exe still not on PATH after vcvars64.
    exit /b 1
)

REM The repo's own convention: UTF-8 everywhere, or the box mangles output.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

%*
exit /b %errorlevel%
