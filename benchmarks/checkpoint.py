"""Resumable result stores for long benchmark runs.

WHY THIS EXISTS
---------------
The evaluation matrix for the paper is hours of GPU time per arm, on a single
desktop card that loses power. A run that keeps its results in memory and writes
one JSON at the end throws away everything it has done when that happens, and
the natural reaction -- rerun it and hope -- is how a matrix silently ends up
containing a mixture of two different code states.

So results are written as they are produced, one JSON object per line, and a
restart skips whatever is already on disk. A power cut costs at most the item
that was in flight.

DESIGN
------
* Append-only JSONL. No rewriting, so a truncated final line is the worst
  corruption possible, and `load_done` discards exactly that line and keeps the
  rest.
* Every record carries a `key` that must be DETERMINISTIC for a given unit of
  work. Resume is `key in done` and nothing more.
* flush + fsync after every record. Slower than buffering, and the whole point:
  the OS buffer is precisely what a power cut discards.
* The run's configuration is stored beside the results in a `.meta.json`. On
  resume the config is compared against it and a MISMATCH IS FATAL -- silently
  appending rows measured under a different preset, quantization, or context
  length to rows measured under another is the failure this whole module exists
  to prevent, and it is invisible in the output table.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, Optional, Set


def _pid_alive(pid: int) -> bool:
    """True if that pid is a live process. Conservative: unknown -> alive."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import subprocess
            out = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=30).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except Exception:                                            # noqa: BLE001
        return False


class ResumableJSONL:
    """Append-only, fsync'd, resumable record store."""

    def __init__(self, path: str, config: Optional[Dict[str, Any]] = None,
                 strict_config: bool = True, read_only: bool = False):
        """read_only=True: read the store without claiming it.

        Scoring and reporting only ever READ. Making them take the writer's
        lock would mean --compare could not be run against a file a live
        campaign owns, and worse, a reader could RECLAIM a lock it has no
        intention of honouring -- re-opening the door this lock exists to shut.
        A reader also must not create the file or its meta sidecar.
        """
        self.path = os.path.abspath(path)
        self.read_only = read_only
        self.meta_path = self.path + ".meta.json"
        self.config = dict(config or {})
        self._fh = None
        self.lock_path = None
        if read_only:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._check_config(strict_config)
        self._acquire_lock()
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    # ── config guard ────────────────────────────────────────────────────────
    def _check_config(self, strict: bool) -> None:
        if not os.path.exists(self.meta_path):
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            return
        with open(self.meta_path, encoding="utf-8") as f:
            old = json.load(f)
        diff = {k: (old.get(k), self.config.get(k))
                for k in set(old) | set(self.config)
                if old.get(k) != self.config.get(k)}
        if diff and strict:
            raise SystemExit(
                f"\nCHECKPOINT CONFIG MISMATCH for {self.path}\n"
                + "\n".join(f"  {k}: on-disk={o!r}  now={n!r}"
                            for k, (o, n) in sorted(diff.items()))
                + "\n\nThe rows already on disk were produced under a different\n"
                  "configuration. Appending to them would build one table out of\n"
                  "two experiments. Point --out at a new file, or delete the old\n"
                  "one if those rows are meant to be discarded.\n")

    # ── single-writer lock ──────────────────────────────────────────────────
    def _acquire_lock(self) -> None:
        """Refuse to open a store another LIVE process is already writing.

        Two harness invocations were once left running against the same output
        file at once -- an orphaned arm plus a restarted campaign. Append+fsync
        meant neither corrupted a line, and load_latest() deduplicates by key,
        so the QUALITY rows survived. What did not survive is everything
        measured while both were resident on one 12 GB card: peak memory and
        latency were recorded under contention, at 96% GPU and 11.8 GB, with
        both processes spilling.

        Nothing in the data marks that. A lock is the only place to catch it.
        """
        self.lock_path = self.path + ".lock"
        if os.path.exists(self.lock_path):
            try:
                with open(self.lock_path, encoding="utf-8") as f:
                    prev = json.load(f)
            except Exception:                                    # noqa: BLE001
                prev = {}
            pid = prev.get("pid")
            if pid and _pid_alive(int(pid)):
                raise SystemExit(
                    "\nANOTHER RUN IS ALREADY WRITING %s\n"
                    "  pid %s, started %s\n\n"
                    "Two processes sharing one output file also share one GPU:\n"
                    "every peak-memory and latency number either of them records\n"
                    "is measured under contention. Stop that process, or point\n"
                    "--out somewhere else.\n"
                    % (self.path, pid, prev.get("started")))
            print("[ckpt] reclaiming stale lock from pid %s (not running)" % pid)
        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(),
                       "started": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

    def _release_lock(self) -> None:
        lp = getattr(self, "lock_path", None)
        if not lp or not os.path.exists(lp):
            return
        # READ FIRST, THEN REMOVE. Windows refuses to delete an open file, so
        # calling os.remove() inside the `with open(...)` block raises
        # PermissionError -- which the blanket except below then swallows,
        # leaving the lock behind. Every subsequent run of the same arm would
        # have died with "another run is already writing", pointing at a pid
        # that had exited cleanly.
        try:
            with open(lp, encoding="utf-8") as f:
                owner = json.load(f).get("pid")
        except Exception:                                        # noqa: BLE001
            owner = None
        if owner is not None and owner != os.getpid():
            return                       # not ours; leave it alone
        try:
            os.remove(lp)
        except Exception:                                        # noqa: BLE001
            pass

    # ── resume ──────────────────────────────────────────────────────────────
    def load_done(self, include_errors: bool = False) -> Set[str]:
        """Keys already recorded. Tolerates a torn final line from a power cut.

        Records carrying an `error` field are NOT counted as done by default.
        An item that failed because the GPU was momentarily out of memory, or
        because the run was killed mid-item, has to be retried on the next pass
        -- treating a failure as a completed unit of work is how a matrix ends
        up with silent holes in it that look like real zeros. The failed row
        stays on disk as a record of what happened; `load_records` still
        returns it, and readers deduplicate by key.
        """
        done: Set[str] = set()
        if not os.path.exists(self.path):
            return done
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Only the LAST line may legitimately be torn (the process died
                # mid-write). Anything earlier means real corruption, and
                # continuing would quietly drop completed work.
                if i == len(lines) - 1:
                    print(f"[ckpt] discarding torn final line in {self.path}")
                    continue
                raise
            if "key" not in rec:
                continue
            if rec.get("error") and not include_errors:
                continue
            done.add(rec["key"])
        return done

    def load_latest(self) -> Dict[str, Dict[str, Any]]:
        """One record per key: the LAST one written wins, so a successful retry
        supersedes the failed attempt that preceded it."""
        latest: Dict[str, Dict[str, Any]] = {}
        for rec in self.load_records():
            k = rec.get("key")
            if k is not None:
                latest[k] = rec
        return latest

    def load_records(self) -> list:
        recs = []
        if not os.path.exists(self.path):
            return recs
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    continue
                raise
        return recs

    # ── write ───────────────────────────────────────────────────────────────
    def append(self, key: str, **fields: Any) -> None:
        if self._fh is None:
            raise RuntimeError("this store was opened read_only; cannot append")
        rec = {"key": key, **fields}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:                                        # noqa: BLE001
            pass
        self._release_lock()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def filter_pending(items: Iterable, key_fn, done: Set[str]):
    """(pending, n_skipped) — items whose key is not already recorded."""
    pending, skipped = [], 0
    for it in items:
        if key_fn(it) in done:
            skipped += 1
        else:
            pending.append(it)
    return pending, skipped
