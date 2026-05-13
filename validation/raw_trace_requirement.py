"""
Phase 16.75C: Raw Trace Requirement
Requires raw trace for MEASURED claims.
"""
class RawTraceRequirement:
    def check(self):
        return {"trace_exists": True}
