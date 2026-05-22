class GenerationEfficiencyTracker:
    """
    Tracks the efficiency of token generation (tokens per Joule or tokens per unit of memory bandwidth).
    Links throughput to physical resource consumption.
    """
    def __init__(self, hardware_power_w=300):
        self.power_w = hardware_power_w

    def calculate_efficiency(self, tps):
        # Tokens per second per Watt (Tokens/Joule)
        return tps / self.power_w
