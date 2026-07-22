import time
import json
from typing import Dict, Any, List
from fastapi import FastAPI, Response

class ServingObservabilityBridge:
    """
    Exposes runtime telemetry and exports serving metrics.
    Provides a Prometheus-compatible /metrics endpoint.
    """
    def __init__(self, scheduler, session_manager, recovery_engine):
        self.scheduler = scheduler
        self.session_manager = session_manager
        self.recovery_engine = recovery_engine
        self.start_time = time.time()

    def get_prometheus_metrics(self) -> str:
        """
        Generates metrics in Prometheus exposition format.
        """
        sched_metrics = self.scheduler.get_serving_metrics()
        rec_metrics = self.recovery_engine.get_recovery_metrics()
        
        uptime = time.time() - self.start_time
        
        lines = [
            f"# HELP dkv_uptime_seconds Total system uptime",
            f"# TYPE dkv_uptime_seconds counter",
            f"dkv_uptime_seconds {uptime}",
            
            f"# HELP dkv_active_sessions Number of currently active sessions",
            f"# TYPE dkv_active_sessions gauge",
            f"dkv_active_sessions {len(self.session_manager.list_sessions())}",
            
            f"# HELP dkv_processed_requests_total Total requests processed",
            f"# TYPE dkv_processed_requests_total counter",
            f"dkv_processed_requests_total {sched_metrics['processed_requests']}",
            
            f"# HELP dkv_average_latency_ms Average request latency in milliseconds",
            f"# TYPE dkv_average_latency_ms gauge",
            f"dkv_average_latency_ms {sched_metrics['average_latency_ms']}",
            
            f"# HELP dkv_recovery_rate_ratio Successful recovery rate (0.0 to 1.0)",
            f"# TYPE dkv_recovery_rate_ratio gauge",
            f"dkv_recovery_rate_ratio {rec_metrics['recovery_rate']}"
        ]
        
        return "\n".join(lines) + "\n"

    def attach_to_app(self, app: FastAPI):
        """
        Mounts the /metrics endpoint to the FastAPI app.
        """
        @app.get("/metrics")
        async def metrics():
            return Response(content=self.get_prometheus_metrics(), media_type="text/plain")
            
        @app.get("/health")
        async def health():
            return {"status": "healthy", "uptime": time.time() - self.start_time}

    def log_operational_diagnostic(self, event_type: str, details: Dict[str, Any]):
        """
        Structured logging for production diagnostics.
        """
        entry = {
            "timestamp": time.time(),
            "event": event_type,
            "details": details
        }
        # In a real system, this might go to a JSON logger or elk stack
        print(f"[OBS] DIAGNOSTIC: {json.dumps(entry)}")
