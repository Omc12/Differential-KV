import json
from pathlib import Path
from typing import Dict, Any

class LPTTraceSystem:
    """
    LPT Trace System
    
    Streams and records exactly the 10 mandated end-to-end reality-grounded LPT trace files.
    """
    def __init__(self, target_dir: Path):
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        self.trace_names = [
            "request_path",
            "session_persistence",
            "kv_lifecycle",
            "replay_participation",
            "dsr_runtime",
            "stream_flush",
            "frontend_emission",
            "visible_tps",
            "conversation_state",
            "live_runtime_alignment"
        ]
        
        self.handles = {}
        for name in self.trace_names:
            file_path = self.target_dir / f"{name}_trace.jsonl"
            self.handles[name] = open(file_path, "w", encoding="utf-8")

    def write_record(self, trace_name: str, record: Dict[str, Any]):
        if trace_name in self.handles:
            self.handles[trace_name].write(json.dumps(record) + "\n")
            self.handles[trace_name].flush()

    def close(self):
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()
        
    def __del__(self):
        self.close()
