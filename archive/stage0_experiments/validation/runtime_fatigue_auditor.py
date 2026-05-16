import numpy as np

class RuntimeFatigueAuditor:
    """
    Analyzes performance decay curves to ensure they are bounded and graceful.
    """
    def __init__(self):
        pass

    def analyze_decay(self, latency_history: list):
        if len(latency_history) < 10:
            return "INSUFFICIENT_DATA"
            
        # Fit linear trend to latency
        x = np.arange(len(latency_history))
        y = np.array(latency_history)
        slope, _ = np.polyfit(x, y, 1)
        
        if slope > 0.01: # Significant increase in latency over time
            return "FATIGUED"
        return "STABLE"
