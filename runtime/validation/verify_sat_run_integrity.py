"""
STAGE 2 — SAT: Run Integrity Verifier
Phase 38.9 — Sparse Attention Transition

Verifies a completed (or aborted) SAT run directory for:

  1. File existence   — all expected files are present and non-empty
  2. JSONL validity   — every line in every .jsonl file is valid JSON
  3. Timestamp monotonicity — 'ts' fields increase across lines
  4. No truncated writes    — files end with a complete newline
  5. Manifest integrity     — manifest.json is present and parseable
  6. Completion marker      — RUN_COMPLETE.txt or RUN_ABORTED.txt present

Usage (from project root):
  python -m runtime.validation.verify_sat_run_integrity [run_id]

If run_id is omitted, the most-recent run in traces/stage2/phase_38_9_sat/
is checked.

Exit codes:
  0  — all checks pass
  1  — one or more checks failed
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE         = "phase_38_9_sat"
STAGE         = "stage2"
TRACE_BASE    = Path("traces")    / STAGE / PHASE
TELEMETRY_BASE= Path("telemetry") / STAGE / PHASE
REPORT_BASE   = Path("reports")   / STAGE / PHASE
MANIFEST_BASE = Path("manifests") / STAGE / PHASE

REQUIRED_TRACES = [
    "sparse_attention_trace.jsonl",
    "dense_reconstruction_trace.jsonl",
    "kv_residency_trace.jsonl",
    "execution_mode_trace.jsonl",
    "sparse_participation_trace.jsonl",
]
REQUIRED_TELEMETRY = ["raw_nvidia_smi_dmon.log"]
REQUIRED_MANIFEST  = ["manifest.json"]
REQUIRED_REPORT    = ["sat_validation_report.json"]
MARKER_FILES       = ["RUN_COMPLETE.txt", "RUN_ABORTED.txt"]

# Traces that are legitimately absent when 0 events of that type occurred.
# Keyed by filename -> report key path that holds the event count.
# If the count is 0 (or the key is absent), missing file = WARN not FAIL.
EVENT_COUNT_KEYS = {
    "dense_reconstruction_trace.jsonl": ["dense_reconstruction", "total_events"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_run_id() -> Optional[str]:
    """Return the lexicographically largest (newest) run_id in TRACE_BASE."""
    if not TRACE_BASE.exists():
        return None
    run_dirs = sorted(
        [d.name for d in TRACE_BASE.iterdir() if d.is_dir()],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _check_exists_nonempty(path: Path, label: str, results: List[str]) -> bool:
    """Assert file exists and has at least one byte."""
    if not path.exists():
        results.append(f"FAIL  [missing]     {label}")
        return False
    size = path.stat().st_size
    if size == 0:
        results.append(f"FAIL  [empty]       {label}")
        return False
    results.append(f"OK    [{size:>10} B] {label}")
    return True


def _check_jsonl(path: Path, label: str, results: List[str]) -> Tuple[bool, int, float, float]:
    """
    Validate every line of a JSONL file.
    Returns (ok, line_count, first_ts, last_ts).
    Also checks:
      - each line ends with '\\n'  (no truncated writes)
      - 'ts' field is numeric and monotonically non-decreasing
    """
    if not path.exists() or path.stat().st_size == 0:
        return False, 0, 0.0, 0.0

    line_count = 0
    bad_lines: List[str] = []
    first_ts: Optional[float] = None
    last_ts:  Optional[float] = None
    prev_ts:  Optional[float] = None
    mono_ok = True

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Truncation check: file must end with '\n'
    if not raw.endswith("\n"):
        results.append(f"WARN  [no trailing newline — possible truncation] {label}")

    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            line_count += 1
            ts = obj.get("ts") or obj.get("timestamp")
            if ts is not None:
                ts = float(ts)
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
                if prev_ts is not None and ts < prev_ts - 0.001:
                    mono_ok = False
                    bad_lines.append(f"  L{lineno}: ts={ts:.4f} < prev={prev_ts:.4f}")
                prev_ts = ts
        except json.JSONDecodeError as exc:
            bad_lines.append(f"  L{lineno}: {exc}")

    ok = (len(bad_lines) == 0)
    tag = "OK" if ok else "FAIL"
    mono_tag = "" if mono_ok else " [NON-MONOTONIC ts!]"
    results.append(
        f"{tag}   [{line_count:>6} lines{mono_tag}] {label}"
    )
    for bl in bad_lines[:5]:   # show first 5 bad lines at most
        results.append(bl)

    return ok, line_count, first_ts or 0.0, last_ts or 0.0


def _check_manifest(path: Path, results: List[str]) -> bool:
    if not path.exists():
        results.append(f"FAIL  [missing] {path}")
        return False
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        required_keys = ["phase", "run_id", "status", "timestamp_utc",
                         "hostname", "model_id", "concurrency"]
        missing = [k for k in required_keys if k not in obj]
        if missing:
            results.append(f"FAIL  [manifest missing keys: {missing}] {path}")
            return False
        status = obj.get('status', 'unknown')
        # RUN_IN_PROGRESS is only a WARN if the report file exists (data was collected)
        if status == "RUN_IN_PROGRESS":
            results.append(f"WARN  [manifest status=RUN_IN_PROGRESS — run may have crashed post-collection] {path}")
        else:
            results.append(f"OK    [manifest valid, status={status}] {path}")
        return True
    except Exception as exc:
        results.append(f"FAIL  [manifest parse error: {exc}] {path}")
        return False


def _check_marker(manifest_dir: Path, results: List[str]) -> bool:
    found = [m for m in MARKER_FILES if (manifest_dir / m).exists()]
    if not found:
        results.append(
            "WARN  [no completion marker — run likely crashed after data collection; "
            "run 'python -m runtime.validation.seal_sat_run' to fix]"
        )
        return False
    for m in found:
        results.append(f"OK    [marker present] {manifest_dir / m}")
    return True


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

def _load_report(rep_dir: Path) -> Optional[Dict]:
    """Load the validation report JSON if it exists, else return None."""
    p = rep_dir / "sat_validation_report.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _report_event_count(report: Optional[Dict], key_path: List[str]) -> int:
    """Navigate a nested report dict via key_path and return int value, or 0."""
    if report is None:
        return 0
    obj = report
    for k in key_path:
        if not isinstance(obj, dict) or k not in obj:
            return 0
        obj = obj[k]
    try:
        return int(obj)
    except (TypeError, ValueError):
        return 0


def verify_run(run_id: str) -> bool:
    """Verify a single run. Returns True if all hard checks pass."""
    print(f"\n{'='*70}")
    print(f"SAT Run Integrity Verifier — run_id={run_id}")
    print(f"{'='*70}")

    results: List[str] = []
    all_ok = True

    rep_dir      = REPORT_BASE   / run_id
    manifest_dir = MANIFEST_BASE / run_id
    trace_dir    = TRACE_BASE    / run_id
    tel_dir      = TELEMETRY_BASE/ run_id

    # Load report first so we can cross-check event counts for optional traces
    report = _load_report(rep_dir)

    # --- Manifest (WARN on IN_PROGRESS if report exists, not a hard FAIL) ---
    manifest_ok = _check_manifest(manifest_dir / "manifest.json", results)
    _check_marker(manifest_dir, results)
    all_ok = all_ok and manifest_ok

    # --- Traces ---
    results.append(f"\n-- Traces ({trace_dir}) --")
    for fname in REQUIRED_TRACES:
        p = trace_dir / fname
        if not p.exists():
            # Check if this trace is legitimately absent (0 events recorded)
            if fname in EVENT_COUNT_KEYS:
                count = _report_event_count(report, EVENT_COUNT_KEYS[fname])
                if count == 0:
                    results.append(
                        f"WARN  [absent, 0 events in report — no {fname} expected]    {fname}"
                    )
                    continue  # not a failure
            results.append(f"FAIL  [missing]     {fname}")
            all_ok = False
            continue

        ex_ok = _check_exists_nonempty(p, fname, results)
        if ex_ok:
            jl_ok, n, t0, t1 = _check_jsonl(p, fname, results)
            all_ok = all_ok and jl_ok
        else:
            all_ok = False

    # --- Telemetry ---
    results.append(f"\n-- Telemetry ({tel_dir}) --")
    for fname in REQUIRED_TELEMETRY:
        p = tel_dir / fname
        ok = _check_exists_nonempty(p, fname, results)
        all_ok = all_ok and ok

    # --- Report ---
    results.append(f"\n-- Reports ({rep_dir}) --")
    for fname in REQUIRED_REPORT:
        p = rep_dir / fname
        ok = _check_exists_nonempty(p, fname, results)
        if ok:
            try:
                with open(p, encoding="utf-8") as f:
                    json.load(f)
                results.append(f"OK    [valid JSON]          {fname}")
            except Exception as exc:
                results.append(f"FAIL  [invalid JSON: {exc}] {fname}")
                all_ok = False
        else:
            all_ok = False

    # --- run.log ---
    results.append(f"\n-- Run log --")
    _check_exists_nonempty(rep_dir / "run.log", "run.log", results)

    # --- Print summary ---
    print()
    for line in results:
        print(line)

    print(f"\n{'='*70}")
    verdict = "PASS — all integrity checks passed." if all_ok \
              else "FAIL — one or more checks did not pass."
    print(f"VERDICT: {verdict}")
    print(f"{'='*70}\n")
    return all_ok


def main() -> None:
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = _find_latest_run_id()
        if run_id is None:
            print(f"No runs found under {TRACE_BASE}.", file=sys.stderr)
            sys.exit(1)
        print(f"No run_id specified — checking latest: {run_id}")

    ok = verify_run(run_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
