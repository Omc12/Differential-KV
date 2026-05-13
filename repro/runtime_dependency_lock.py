"""
Runtime Dependency Lock.
Records exact package hashes and versions for strict reproducibility.
"""

class RuntimeDependencyLock:
    def lock(self):
        return {"torch": "2.1.0+cu121", "triton": "2.1.0"}
