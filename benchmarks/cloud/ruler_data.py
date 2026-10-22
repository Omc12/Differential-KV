"""Generate the official RULER data on any machine, and prove it is the same data.

benchmarks/generate_ruler_data.sh is tied to this Windows box (interpreter
path, %TEMP% work copy). This is the same generation -- the vendored
ruler_official/data/prepare.py, one call per (length, task), the same seed and
sample count -- from Python, so a Linux worker can build what its job reads.

SAME ITEMS EVERYWHERE OR THE PAIRS MEAN NOTHING
    Paired bootstraps compare two arms on the SAME items. Arms of one store may
    run on different machines, so every machine must generate byte-identical
    files. Each generated file's sha256 is checked against
    ruler_data_sha256.json, recorded from the data the paper already used; a
    file with no recorded hash is recorded by the worker in done.json and
    `ctl.py status` flags any (model, length, task) whose machines disagree.
    The draws depend on the sample count, so it is always RULER_NSAMP (20),
    whatever a job reads.

    python benchmarks/cloud/ruler_data.py ensure --model Qwen/Qwen3.5-4B --length 8192
    python benchmarks/cloud/ruler_data.py hashes   # rewrite ruler_data_sha256.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from jobs import RULER_13, RULER_NSAMP  # noqa: E402

OFFICIAL = os.path.join(REPO, "benchmarks", "ruler_official")
HASHES = os.path.join(HERE, "ruler_data_sha256.json")
DEFAULT_ROOT = os.environ.get("DKV_RULER_DATA") or os.path.join(REPO, "paper", "results", "ruler_data")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _count(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _work_copy() -> str:
    """prepare.py opens ../synthetic.yaml and the essays relative to itself; a
    space-free copy also keeps RULER's own scripts away from quoting bugs."""
    work = os.path.join(tempfile.gettempdir(), "dkv_ruler_work")
    if not (os.path.isfile(os.path.join(work, "data", "prepare.py"))
            and os.path.isfile(os.path.join(work, "synthetic.yaml"))):
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(OFFICIAL, work, ignore=shutil.ignore_patterns("__pycache__"))
    return work


class _Lock:
    """One generator per (model, length) per machine; others wait for it."""

    def __init__(self, path: str, timeout_s: float = 6 * 3600):
        self.path, self.timeout = path, timeout_s

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - os.path.getmtime(self.path) > self.timeout:
                    os.remove(self.path)       # holder died; generation restarts
                    continue
                time.sleep(10)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def ensure(model: str, length: int, root: str = DEFAULT_ROOT, verbose: bool = True) -> Dict[str, str]:
    """Make root/<tag>/<length>/<task>/validation.jsonl exist for all 13 tasks
    with RULER_NSAMP items. Returns {task: sha256}."""
    tag = model.split("/")[-1]
    dest = os.path.join(root, tag, str(length))
    os.makedirs(dest, exist_ok=True)

    def missing():
        return [t for t in RULER_13
                if _count(os.path.join(dest, t, "validation.jsonl")) < RULER_NSAMP]

    if missing():
        with _Lock(os.path.join(root, tag, f".gen_{length}.lock")):
            todo = missing()
            work = _work_copy() if todo else ""
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            # prepare.py launches each generator with sys.executable already;
            # PATH first is for anything RULER spawns by name.
            env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
            for t in todo:
                t0 = time.time()
                f = os.path.join(dest, t, "validation.jsonl")
                if os.path.exists(f):
                    os.remove(f)          # a short file is a dead run: regenerate whole
                p = subprocess.run(
                    [sys.executable, "prepare.py", "--save_dir", dest,
                     "--benchmark", "synthetic", "--task", t,
                     "--tokenizer_path", model, "--tokenizer_type", "hf",
                     "--max_seq_length", str(length), "--num_samples", str(RULER_NSAMP),
                     "--model_template_type", "base"],
                    cwd=os.path.join(work, "data"), env=env, capture_output=True,
                    text=True, encoding="utf-8", errors="replace")
                n = _count(f)
                if verbose:
                    print(f"[ruler-data] {tag} {t} @ {length}: {n} items "
                          f"({time.time() - t0:.0f} s)", flush=True)
                if n < RULER_NSAMP:
                    raise RuntimeError(
                        f"RULER generation failed for {tag} {t} @ {length} ({n} items):\n"
                        + (p.stderr or p.stdout)[-2000:])
    return {t: sha256(os.path.join(dest, t, "validation.jsonl")) for t in RULER_13}


def verify(model: str, length: int, shas: Dict[str, str]) -> Dict[str, str]:
    """{task: 'match' | 'MISMATCH' | 'new'} against the recorded hashes."""
    ref = (_load_hashes().get(model.split("/")[-1]) or {}).get(str(length)) or {}
    return {t: ("new" if t not in ref else "match" if ref[t] == s else "MISMATCH")
            for t, s in shas.items()}


def _load_hashes() -> Dict:
    try:
        with open(HASHES, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_hashes(root: str) -> Dict:
    """Hash every complete (model, length) set under root, 20 items or more."""
    out: Dict = {}
    for tag in sorted(os.listdir(root)):
        for L in sorted(os.listdir(os.path.join(root, tag)), key=lambda x: int(x) if x.isdigit() else 0):
            d = os.path.join(root, tag, L)
            if not L.isdigit():
                continue
            files = {t: os.path.join(d, t, "validation.jsonl") for t in RULER_13}
            if all(_count(f) == RULER_NSAMP for f in files.values()):
                out.setdefault(tag, {})[L] = {t: sha256(f) for t, f in files.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ensure", "hashes", "verify"])
    ap.add_argument("--model", default="")
    ap.add_argument("--length", type=int, default=0)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    a = ap.parse_args()
    if a.cmd == "hashes":
        h = record_hashes(a.root)
        with open(HASHES, "w", encoding="utf-8", newline="\n") as f:
            json.dump(h, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"wrote {HASHES}: " + ", ".join(f"{t} {sorted(map(int, v))}" for t, v in h.items()))
        return
    shas = ensure(a.model, a.length, a.root)
    v = verify(a.model, a.length, shas)
    print(json.dumps(v, indent=1))
    if "MISMATCH" in v.values():
        raise SystemExit("generated data differs from the paper's items")


if __name__ == "__main__":
    main()
