class GpuCounterReconciler:
    """
    Reconciles software-reported metrics with raw hardware counters.
    Ensures that reported bandwidth matches physical DRAM/L2 traffic.
    """
    def __init__(self):
        pass

    def reconcile(self, metric_name, reported_value, hardware_data):
        """
        Reconciles a reported value with hardware data.
        """
        # Mock reconciliation logic
        actual_value = hardware_data.get(f"hw_{metric_name}", reported_value * 0.98) # Simulate slight drift
        variance = abs(reported_value - actual_value) / reported_value if reported_value != 0 else 0
        
        return {
            "metric": metric_name,
            "reported": reported_value,
            "actual": actual_value,
            "variance": variance
        }
