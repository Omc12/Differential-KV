class TraceGroundingEngine:
    """
    Main engine for grounding all performance claims in physical hardware traces.
    Reconciles occupancy, bandwidth, and latency against hardware counters.
    """
    def __init__(self, validator, reconciler):
        self.validator = validator
        self.reconciler = reconciler

    def ground_claim(self, claim_type, claimed_value, trace_data):
        """
        Grounds a specific performance claim.
        Example: claim_type='occupancy', claimed_value=0.85
        """
        validation = self.validator.validate_trace(trace_data.get("trace_path", ""))
        
        if not validation["valid"]:
            return {"grounded": False, "reason": "Invalid trace"}
            
        # Reconcile against actual hardware counters
        reconciliation = self.reconciler.reconcile(claim_type, claimed_value, trace_data)
        
        return {
            "claim_type": claim_type,
            "claimed_value": claimed_value,
            "actual_value": reconciliation["actual"],
            "variance": reconciliation["variance"],
            "grounded": reconciliation["variance"] < 0.05 # 5% tolerance
        }
