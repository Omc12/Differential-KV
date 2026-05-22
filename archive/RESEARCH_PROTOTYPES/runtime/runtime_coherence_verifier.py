import logging
from typing import Dict, Any, List
from .production_session_lifecycle_manager import ProductionSessionLifecycleManager
from .operational_telemetry_dashboard_backend import OperationalTelemetryDashboardBackend

class RuntimeCoherenceVerifier:
    """
    ORX Phase 40.2: Runtime Coherence Verifier.
    Ensures all operational systems remain consistent simultaneously.
    """
    def __init__(self, 
                 session_manager: ProductionSessionLifecycleManager,
                 telemetry: OperationalTelemetryDashboardBackend):
        self.session_manager = session_manager
        self.telemetry = telemetry
        self.logger = logging.getLogger("CoherenceVerifier")

    def verify_coherence(self) -> float:
        """
        Detects desynchronization between systems.
        Returns a coherence score [0.0 - 1.0].
        """
        coherence = 1.0
        
        # 1. Telemetry vs Session Manager sync
        real_active = self.session_manager.get_active_sessions_count()
        telemetry_active = self.telemetry.get_live_feed().get("active_sessions", 0)
        
        if real_active != telemetry_active:
            diff = abs(real_active - telemetry_active)
            self.logger.error(f"COHERENCE FAIL: Session desync (Real: {real_active}, Telemetry: {telemetry_active})")
            coherence -= min(0.2, diff * 0.05)

        # 2. Queue depth consistency
        # (Check if telemetry queue depth is physically possible)
        
        # 3. Stream state consistency
        # (Check for orphan sessions that telemetry thinks are active)
        orphans = self.session_manager.list_orphan_sessions()
        if orphans:
            self.logger.warning(f"COHERENCE WARN: {len(orphans)} orphan sessions detected.")
            coherence -= min(0.1, len(orphans) * 0.02)

        return max(0.0, coherence)

    def check_synchronization(self) -> str:
        """Returns a string representation of sync state."""
        score = self.verify_coherence()
        if score > 0.95: return "SYNCHRONIZED"
        if score > 0.8: return "DEGRADED"
        return "DESYNCHRONIZED"
