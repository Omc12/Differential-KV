import torch

class SparseBalanceGuardian:
    """
    PHASE 19.0E: Sparse Balance Guardian.
    Enforces strict memory and compute bounds, shutting down bridge 
    logic if it threatens system stability.
    """
    def __init__(self, tps_floor: float = 10.0, vram_limit_gb: float = 8.0):
        self.tps_floor = tps_floor
        self.vram_limit_gb = vram_limit_gb
        self.emergency_shutdown = False

    def check_stability(self, current_tps: float, current_vram_gb: float) -> bool:
        """
        Returns False if stability is threatened.
        """
        if current_tps < self.tps_floor:
            self.emergency_shutdown = True
            return False
            
        if current_vram_gb > self.vram_limit_gb:
            self.emergency_shutdown = True
            return False
            
        return True

    def should_bypass_bridges(self) -> bool:
        return self.emergency_shutdown
