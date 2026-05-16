import subprocess
import os

class RuntimeDependencyLock:
    """
    Phase 18F: Generates a freeze of all dependencies for reproducibility.
    """
    def __init__(self, export_path: str = "results/reconstruction_18/requirements_lock.txt"):
        self.export_path = export_path

    def lock(self):
        print(f"[PHASE 18F] Locking dependencies to {self.export_path}")
        try:
            with open(self.export_path, 'w') as f:
                subprocess.run(["pip", "freeze"], stdout=f, check=True)
            return True
        except Exception as e:
            print(f"[ERROR] Dependency lock failed: {e}")
            return False

if __name__ == "__main__":
    rdl = RuntimeDependencyLock()
    rdl.lock()
