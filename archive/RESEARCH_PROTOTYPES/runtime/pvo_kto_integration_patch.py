"""
runtime/pvo_kto_integration_patch.py

Unified stabilization orchestrator for PVO + KTO integration.
Ensures that profiling and tuning work together without breaking determinism.
"""

import logging
from typing import Any, Dict

from hardware_materialization.replay_safe_tuning_controller import ReplaySafeTuningController
from hardware_materialization.profiler_epoch_synchronizer import ProfilerEpochSynchronizer
from hardware_materialization.occupancy_hysteresis_stabilizer import OccupancyHysteresisStabilizer
from hardware_materialization.deterministic_microbatch_controller import DeterministicMicrobatchController

logger = logging.getLogger("IntegrationPatch")

class PVOKTOIntegrationPatch:
    """
    Stabilizes the interaction between profiling (PVO) and tuning (KTO).
    """
    def __init__(self, pvo_resolver: Any, kto_resolver: Any):
        self.pvo = pvo_resolver
        self.kto = kto_resolver
        
        # Patch Components
        self.tuning_controller = ReplaySafeTuningController()
        self.epoch_sync = ProfilerEpochSynchronizer()
        self.hysteresis = OccupancyHysteresisStabilizer()
        self.microbatch_ctrl = DeterministicMicrobatchController()

    def synchronize_tuning_epoch(self, new_config: Dict[str, Any]):
        """
        Coordinates a tuning update across both profiling and execution layers.
        """
        # 1. Start a new telemetry epoch
        self.epoch_sync.start_epoch(new_config)
        
        # 2. Lock microbatch size if we're in a stable replay phase
        if "microbatch_size" in new_config:
            self.microbatch_ctrl.lock_microbatch_size(new_config["microbatch_size"])
            
        logger.info("Synchronized PVO/KTO tuning epoch.")

    def safe_apply_tuning(self, key: str, current_occupancy: float, tuning_fn: callable):
        """
        Applies tuning only if it's safe (no oscillation, no replay drift).
        """
        # 1. Check Hysteresis
        if not self.hysteresis.should_retune(key, current_occupancy):
            return False
            
        # 2. Check Replay Safety
        if not self.tuning_controller.request_tuning_window(key):
            return False
            
        # 3. Apply Tuning
        tuning_fn()
        
        # 4. Record Success
        self.hysteresis.record_tuning(key, current_occupancy)
        return True

    def get_stability_metrics(self) -> Dict[str, Any]:
        """Returns metrics for integration stability."""
        return {
            "current_epoch": self.epoch_sync.current_epoch,
            "microbatch_locked": self.microbatch_ctrl.is_locked,
            "occupancy_stability": 1.0 # Simplified stability index
        }
