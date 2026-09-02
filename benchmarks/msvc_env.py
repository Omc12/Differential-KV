"""Put the MSVC toolchain on PATH from inside Python, before torch.compile runs.

WHY THIS EXISTS
---------------
torch.compile / Inductor needs a host C++ compiler. On Windows that is cl.exe,
which MSVC only puts on PATH inside a "Developer Command Prompt" -- installing
Build Tools is not enough. Without it DKV logs

    [DKV JIT] _reconstruct_and_score: backend failed at call time
    (RuntimeError: Compiler: cl is not found.) Falling back to eager

and every latency number after that UNDERSTATES DKV, because the
Inductor-wrapped reconstruction runs unfused. The fallback is partial rather
than total -- the Triton fused decode kernel still compiles and runs -- so the
symptom is a quiet slowdown, not an error, and it is easy to publish without
noticing.

tools/with_msvc.bat solves this for a command line, but every benchmark
invocation would then have to be routed through cmd.exe, and the harnesses take
arguments like '{"budget": 2016}' that do not survive that quoting intact. So
the environment is imported in-process instead: same vcvars64.bat, same
variables, no shell in the middle.

MUST BE CALLED BEFORE the first torch.compile. Calling it at the top of main()
is enough -- Inductor reads the environment when it compiles, not at import.
"""

from __future__ import annotations

import os
import shutil
import subprocess


_VSWHERE = os.path.join(
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    "Microsoft Visual Studio", "Installer", "vswhere.exe")

# Only these are taken from the developer environment. Importing everything
# vcvars sets would drag in unrelated state and could clobber variables the
# benchmark itself relies on.
_WANTED = ("PATH", "INCLUDE", "LIB", "LIBPATH", "VCINSTALLDIR",
           "VCToolsInstallDir", "WindowsSdkDir", "WindowsSDKVersion",
           "UCRTVersion", "VSINSTALLDIR")


def _find_vcvars() -> str:
    if os.path.exists(_VSWHERE):
        try:
            out = subprocess.run(
                [_VSWHERE, "-latest", "-products", "*", "-property", "installationPath"],
                capture_output=True, text=True, timeout=60).stdout.strip()
            if out:
                cand = os.path.join(out, "VC", "Auxiliary", "Build", "vcvars64.bat")
                if os.path.exists(cand):
                    return cand
        except Exception:                                        # noqa: BLE001
            pass
    for root in (r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"C:\Program Files\Microsoft Visual Studio"):
        for year in ("2022", "2019"):
            for ed in ("BuildTools", "Community", "Professional", "Enterprise"):
                cand = os.path.join(root, year, ed, "VC", "Auxiliary", "Build",
                                    "vcvars64.bat")
                if os.path.exists(cand):
                    return cand
    return ""


def ensure_msvc(verbose: bool = True) -> bool:
    """Make cl.exe available to this process. Returns True if it is."""
    if os.name != "nt":
        return True
    if shutil.which("cl"):
        return True

    vcvars = _find_vcvars()
    if not vcvars:
        if verbose:
            print("[msvc] no vcvars64.bat found — Inductor will fall back to "
                  "eager and TIMINGS WILL UNDERSTATE DKV. Label them "
                  "eager-path if you report them.", flush=True)
        return False

    try:
        # `set` after vcvars gives the fully resolved developer environment.
        out = subprocess.run(f'"{vcvars}" >nul 2>&1 && set', shell=True,
                             capture_output=True, text=True, timeout=180).stdout
    except Exception as e:                                       # noqa: BLE001
        if verbose:
            print(f"[msvc] vcvars failed ({type(e).__name__}); staying on eager.",
                  flush=True)
        return False

    n = 0
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k in _WANTED:
            os.environ[k] = v
            n += 1

    ok = shutil.which("cl") is not None
    if verbose:
        if ok:
            print(f"[msvc] developer environment imported ({n} vars); "
                  f"cl.exe at {shutil.which('cl')}", flush=True)
        else:
            print("[msvc] imported vcvars but cl.exe still not resolvable — "
                  "timings will be eager-path.", flush=True)
    return ok
