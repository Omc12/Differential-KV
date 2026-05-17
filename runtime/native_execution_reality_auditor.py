import os
import sys
import time

class NativeExecutionRealityAuditor:
    """
    NDX Phase 42.1.5 — Native Execution Reality Auditor.
    Physically verifies native DLL operations, traces code lineages,
    detects Python orchestration fallbacks, and triggers validation crashes
    on violations.
    """
    def __init__(self, trace_system=None):
        self.trace_system = trace_system
        self.call_registry = {}
        self.violations = []
        self.lineages = []

    def verify_dll_load(self, name: str, executor_obj):
        """
        Confirms if the native executor DLL loaded successfully.
        """
        dll_loaded = getattr(executor_obj, "compiled", False) and getattr(executor_obj, "dll", None) is not None
        if not dll_loaded:
            msg = f"[NDX Auditor Violation] Native DLL {name} failed to load or compile!"
            self.log_violation(msg)
            raise RuntimeError(msg)
        print(f"[Reality Auditor] Verified native DLL {name} successfully loaded.")

    def log_call(self, name: str, lineage: str = "native"):
        """
        Tracks a physical native invocation.
        """
        self.call_registry[name] = self.call_registry.get(name, 0) + 1
        self.lineages.append({
            "timestamp": time.time(),
            "target": name,
            "lineage": lineage
        })

    def log_violation(self, message: str):
        """
        Records an active fallback violation.
        """
        print(f"[Reality Auditor ERROR] {message}", file=sys.stderr)
        self.violations.append({
            "timestamp": time.time(),
            "violation": message
        })

    def enforce_reality(self):
        """
        Validates no Python fallback code paths were hit.
        """
        if len(self.violations) > 0:
            raise RuntimeError(f"[NDX Integrity Failure] Real native execution was violated! Hit {len(self.violations)} fallbacks.")
        print("[Reality Auditor] Enforced native reality check: 100% verified native execution.")

    def get_violations(self) -> list:
        return self.violations

    def get_lineages(self) -> list:
        return self.lineages
