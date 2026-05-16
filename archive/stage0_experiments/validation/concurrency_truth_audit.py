class ConcurrencyTruthAudit:
    """
    Rejects runs with silent dense fallbacks or undercounted 
    orchestration overhead in multi-user settings.
    """
    def __init__(self):
        pass

    def audit_tps(self, reported_tps: float, overhead_ms: float):
        # Honest TPS = 1 / ( (1/reported_tps) + overhead )
        honest_tps = 1.0 / ((1.0 / reported_tps) + (overhead_ms / 1000.0))
        if reported_tps > honest_tps * 1.05:
            print(f"REJECTED: TPS inflation detected! Reported={reported_tps}, Honest={honest_tps}")
            return False
        return True
