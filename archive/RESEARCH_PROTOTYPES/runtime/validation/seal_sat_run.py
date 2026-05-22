"""
SAT Run Sealer — Phase 38.9
Retroactively seals a run that crashed after data collection
but before writing RUN_COMPLETE.txt / updating the manifest.

Usage (from project root):
  python -m runtime.validation.seal_sat_run [run_id]

If run_id is omitted, seals the most-recent run.

Safety checks before sealing:
  - run must have a valid sat_validation_report.json
  - run must NOT already have a RUN_COMPLETE or RUN_ABORTED marker
  - manifest must exist and be parseable
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PHASE         = "phase_38_9_sat"
STAGE         = "stage2"
TRACE_BASE    = Path("traces")    / STAGE / PHASE
REPORT_BASE   = Path("reports")   / STAGE / PHASE
MANIFEST_BASE = Path("manifests") / STAGE / PHASE
MARKER_FILES  = ["RUN_COMPLETE.txt", "RUN_ABORTED.txt"]


def _find_latest_run_id() -> Optional[str]:
    if not TRACE_BASE.exists():
        return None
    run_dirs = sorted([d.name for d in TRACE_BASE.iterdir() if d.is_dir()], reverse=True)
    return run_dirs[0] if run_dirs else None


def seal_run(run_id: str) -> bool:
    print(f"\nSealing run: {run_id}")

    manifest_dir = MANIFEST_BASE / run_id
    rep_dir      = REPORT_BASE   / run_id
    manifest_path = manifest_dir / "manifest.json"
    report_path   = rep_dir / "sat_validation_report.json"

    # --- Safety: must have a valid report ---
    if not report_path.exists():
        print(f"ABORT: no sat_validation_report.json found at {report_path}")
        print("       Cannot seal a run with no collected data.")
        return False

    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        print(f"  Report OK ({report_path.stat().st_size} B)")
    except Exception as exc:
        print(f"ABORT: report is not valid JSON: {exc}")
        return False

    # --- Safety: must not already be sealed ---
    for marker in MARKER_FILES:
        if (manifest_dir / marker).exists():
            print(f"ABORT: run is already sealed ({marker} exists). Nothing to do.")
            return False

    # --- Safety: manifest must exist ---
    if not manifest_path.exists():
        print(f"ABORT: manifest.json not found at {manifest_path}")
        return False

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        print(f"ABORT: manifest.json is not valid JSON: {exc}")
        return False

    # --- Update manifest status ---
    now = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "RUN_COMPLETE"
    manifest["sealed_at"] = now
    manifest["seal_note"] = (
        "Sealed retroactively by seal_sat_run.py — run crashed after "
        "data collection but before writing completion marker."
    )
    manifest["summary"] = report  # embed the report for completeness

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    print(f"  manifest.json updated -> status=RUN_COMPLETE")

    # --- Write completion marker ---
    marker_path = manifest_dir / "RUN_COMPLETE.txt"
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"RUN_COMPLETE\n")
        f.write(f"run_id={run_id}\n")
        f.write(f"ts={now}\n")
        f.write(f"sealed_by=seal_sat_run.py\n")
        f.write(f"note=Data fully collected; sealed post-crash.\n")
    print(f"  RUN_COMPLETE.txt written")

    # --- Append note to run.log if it exists ---
    run_log = rep_dir / "run.log"
    if run_log.exists():
        with open(run_log, "a", encoding="utf-8") as f:
            f.write(f"[{now}] Run sealed retroactively by seal_sat_run.py\n")
        print(f"  run.log updated")

    print(f"\nRun {run_id} sealed successfully.")
    print(f"Re-run verify to confirm: python -m runtime.validation.verify_sat_run_integrity {run_id}")
    return True


def main() -> None:
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = _find_latest_run_id()
        if run_id is None:
            print(f"No runs found under {TRACE_BASE}.", file=sys.stderr)
            sys.exit(1)
        print(f"No run_id specified — sealing latest: {run_id}")

    ok = seal_run(run_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
