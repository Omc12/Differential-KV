import os
import json
import time
import logging
from typing import Dict, List, Any

class PersistentObservabilityLayer:
    """
    Implements persistent runtime logs, long-horizon serving telemetry, 
    and exportable operational reports.
    """
    def __init__(self, log_dir: str = "./operational_telemetry"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.telemetry_history_path = os.path.join(log_dir, "serving_history.jsonl")
        self.event_log_path = os.path.join(log_dir, "operational_events.log")
        
        # Setup specific logger for operational events
        self.op_logger = logging.getLogger("OperationalEvents")
        handler = logging.FileHandler(self.event_log_path)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.op_logger.addHandler(handler)
        self.op_logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: str, severity: str = "INFO"):
        msg = f"[{event_type}] {details}"
        if severity == "INFO":
            self.op_logger.info(msg)
        elif severity == "WARNING":
            self.op_logger.warning(msg)
        elif severity == "ERROR":
            self.op_logger.error(msg)

    def record_telemetry_snapshot(self, metrics: Dict[str, Any]):
        """
        Appends a telemetry snapshot to the persistent history file.
        """
        snapshot = {
            "timestamp": time.time(),
            "metrics": metrics
        }
        with open(self.telemetry_history_path, 'a') as f:
            f.write(json.dumps(snapshot) + "\n")

    def get_serving_history(self, last_n_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Retrieves telemetry history for analysis.
        """
        if not os.path.exists(self.telemetry_history_path):
            return []
            
        history = []
        now = time.time()
        cutoff = now - (last_n_hours * 3600)
        
        with open(self.telemetry_history_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry["timestamp"] >= cutoff:
                        history.append(entry)
                except:
                    continue
        return history

    def generate_operational_report(self) -> Dict[str, Any]:
        """
        Generates an exportable operational report based on persistent data.
        """
        history = self.get_serving_history()
        if not history:
            return {"status": "NO_DATA"}
            
        tps_values = [h["metrics"].get("sustained_tps", 0) for h in history]
        latency_values = [h["metrics"].get("avg_latency_ms", 0) for h in history]
        
        import numpy as np
        return {
            "period_start": history[0]["timestamp"],
            "period_end": history[-1]["timestamp"],
            "sample_count": len(history),
            "avg_tps": float(np.mean(tps_list)) if tps_values else 0,
            "max_tps": float(max(tps_values)) if tps_values else 0,
            "avg_latency": float(np.mean(latency_values)) if latency_values else 0,
            "p95_latency": float(np.percentile(latency_values, 95)) if latency_values else 0
        }
