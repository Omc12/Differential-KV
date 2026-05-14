from validation.reproducibility_bundle_exporter import ReproducibilityBundleExporter
from repro.runtime_dependency_lock import RuntimeDependencyLock
from repro.environment_snapshot import EnvironmentSnapshot
import os

def export_all():
    run_id = "phase18_reconstruction"
    print(f"=== Exporting Phase 18 Reproducibility Bundle [{run_id}] ===")
    
    # 1. Lock dependencies
    rdl = RuntimeDependencyLock()
    rdl.lock()
    
    # 2. Capture environment
    es = EnvironmentSnapshot()
    es.capture()
    
    # 3. Create bundle
    exporter = ReproducibilityBundleExporter(run_id)
    
    # Add critical files
    exporter.add_artifact("results/reconstruction_18/runtime_manifest.json")
    exporter.add_artifact("results/reconstruction_18/environment_snapshot.json")
    exporter.add_artifact("results/reconstruction_18/requirements_lock.txt")
    exporter.add_artifact("results/reconstruction_18/bench_results.json")
    exporter.add_artifact("results/reconstruction_18/reconstruction_18_real_model_tps.md")
    
    bundle_path = exporter.export_bundle()
    print(f"Bundle ready at: {bundle_path}")

if __name__ == "__main__":
    export_all()
