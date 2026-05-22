class PhysicalPlausibilityChecker:
    """
    Checks if reported metrics are physically plausible for the given hardware and model.
    Rejects values like 16M TPS on a single node.
    """
    def __init__(self, hardware_specs=None):
        self.specs = hardware_specs or {"bandwidth_gb_s": 2000} # e.g. H100

    def check_tps(self, tps, model_params_b, bits_per_weight=16):
        # Memory-bound generation: TPS * ModelSize * BytesPerWeight <= Bandwidth
        # This is a rough lower bound for plausibility
        bandwidth_required = (tps * model_params_b * (bits_per_weight / 8)) / 1e9
        
        if bandwidth_required > self.specs["bandwidth_gb_s"] * 1.5: # 50% headroom for architecture
            return False, f"TPS {tps} implies {bandwidth_required:.1f} GB/s, exceeding hardware {self.specs['bandwidth_gb_s']} GB/s"
        return True, "OK"
