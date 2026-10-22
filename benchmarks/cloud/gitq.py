"""A work queue whose only server is a git branch.

WHY GIT
-------
Every machine already needs GitHub for the code, and a push is atomic: it
lands only if the branch has not moved since the pusher last saw it. That is
compare-and-swap, which is all a queue needs. Two machines claiming the same
job both write jobs/<id>/claim.json; the first push lands, the second is
rejected, re-reads the branch, finds the claim taken and picks another job. No
coordinator to keep alive, no credentials beyond one GitHub token, and the
whole campaign state is a directory anyone can `git log`.

LAYOUT OF THE RESULTS BRANCH (an orphan branch: no code history)
    control.json            what the operator wants (see ctl.py)
    machines/<worker>.json  each worker's last heartbeat, GPU, preflight
    jobs/<id>/claim.json    who runs it, since when, last heartbeat, attempt
    jobs/<id>/done.json     how it ended (status, rc, wall time, GPU, code)
    jobs/<id>/out/<path>    its output files, pushed while it runs and at the end
    jobs/<id>/log.txt       the tail of its stdout+stderr

RETRY DISCIPLINE
    Every write is a `transact(fn)`: reset to the remote tip, let fn decide
    from that fresh state and write files, commit, push. A rejected push means
    someone else moved the branch, so the whole thing repeats from the new
    tip. fn must therefore be idempotent and decide only from what it reads --
    which is what makes the claim race safe.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

STALE_MIN = 40          # a claim with no heartbeat for this long is up for grabs
MAX_ATTEMPTS = 2        # a job that failed this many times waits for `ctl retry`


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def age_min(stamp: Optional[str]) -> float:
    if not stamp:
        return 1e9
    t = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 60


def _read(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


class Conflict(Exception):
    """The branch moved under us; transact() retries."""


class GitQueue:
    def __init__(self, url: str, branch: str, workdir: str, who: str = "worker"):
        self.url, self.branch, self.dir, self.who = url, branch, workdir, who

    # ------------------------------------------------------------- git ---
    def git(self, *args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
        p = subprocess.run(["git", "-C", self.dir] + list(args), capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if check and p.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed ({p.returncode}): "
                               f"{(p.stderr or p.stdout).strip()[-600:]}")
        return p

    def open(self) -> "GitQueue":
        if not os.path.isdir(os.path.join(self.dir, ".git")):
            os.makedirs(os.path.dirname(os.path.abspath(self.dir)) or ".", exist_ok=True)
            p = subprocess.run(["git", "clone", "--quiet", "--single-branch", "--branch",
                                self.branch, self.url, self.dir],
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"cannot clone {self.branch} from {self.url}: "
                                   f"{p.stderr.strip()[-400:]}\n(run `ctl.py init` once)")
        self.git("config", "user.name", f"dkv-cloud {self.who}")
        self.git("config", "user.email", "dkv-cloud@users.noreply.github.com")
        self.git("config", "core.autocrlf", "false")
        self.sync()
        return self

    def sync(self) -> None:
        for i in range(6):
            p = self.git("fetch", "--quiet", "origin", self.branch, check=False)
            if p.returncode == 0:
                break
            time.sleep(min(60, 5 * 2 ** i))     # GitHub hiccups: back off, don't die
        else:
            raise RuntimeError("fetch kept failing: " + p.stderr.strip()[-300:])
        self.git("reset", "--quiet", "--hard", "origin/" + self.branch)
        self.git("clean", "-qfdx")

    def transact(self, fn: Callable[[str], Any], message: str, tries: int = 40) -> Any:
        """Run fn(dir) on a fresh tip and push what it wrote; retry on a race.
        fn returns any value; it is handed back once its writes are on the
        remote (or immediately if it wrote nothing)."""
        for i in range(tries):
            self.sync()
            result = fn(self.dir)
            self.git("add", "-A")
            if not self.git("status", "--porcelain").stdout.strip():
                return result
            self.git("commit", "--quiet", "-m", message)
            p = self.git("push", "--quiet", "origin", "HEAD:" + self.branch, check=False)
            if p.returncode == 0:
                return result
            # A race or a network blip retries; a bad token never gets better.
            err = (p.stderr or "").lower()
            if any(s in err for s in ("authentication", "permission", "403", "denied")):
                raise RuntimeError("push not authorized: " + p.stderr.strip()[-400:])
            time.sleep(random.uniform(0.5, 3.0) * min(i + 1, 8))
        raise RuntimeError(f"gave up after {tries} push races: {message}")

    # ------------------------------------------------------------ state ---
    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    def control(self) -> Dict[str, Any]:
        return _read(self.path("control.json")) or {}

    def job_state(self, jid: str, root: Optional[str] = None) -> Dict[str, Any]:
        root = root or self.dir
        return {"claim": _read(os.path.join(root, "jobs", jid, "claim.json")),
                "done": _read(os.path.join(root, "jobs", jid, "done.json"))}

    def all_states(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        jd = self.path("jobs")
        if os.path.isdir(jd):
            for jid in os.listdir(jd):
                out[jid] = self.job_state(jid)
        return out


def classify(st: Dict[str, Any]) -> str:
    """ok | failed | dead (failed too often) | running | stale | pending"""
    done, claim = st.get("done"), st.get("claim")
    if done and done.get("status") == "ok":
        return "ok"
    attempts = (claim or {}).get("attempt", 0)
    if done and done.get("status") != "ok":
        if done.get("released"):
            return "pending"
        return "dead" if attempts >= MAX_ATTEMPTS else "failed"
    if claim and not claim.get("released"):
        return "stale" if age_min(claim.get("heartbeat")) > STALE_MIN else "running"
    return "pending"


def expand(root: str, rel_paths):
    """Output entries to files: an entry ending in "/" is a directory whose
    *.json / *.jsonl files count (decode_fidelity writes one file per arm,
    and a large .pt control that is rebuilt rather than shipped)."""
    out = []
    for rel in rel_paths:
        if rel.endswith("/"):
            d = os.path.join(root, rel)
            if os.path.isdir(d):
                out += [rel + f for f in sorted(os.listdir(d))
                        if f.endswith((".json", ".jsonl"))]
        else:
            out.append(rel)
    return out


def copy_into(src_root: str, rel_paths, dst_root: str) -> int:
    """Copy repo-relative files that exist under src_root to dst_root."""
    n = 0
    for rel in expand(src_root, rel_paths):
        s = os.path.join(src_root, rel)
        if os.path.isfile(s):
            d = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copyfile(s, d)
            n += 1
    return n
