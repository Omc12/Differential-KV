import json
from pathlib import Path
import time

class FSETraceSystem:
    """
    Persists exactly the 10 required FSE traces reflecting real human-visible pacing.
    """
    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.traces = {
            "sentence_group": "sentence_group_trace.jsonl",
            "flush_cadence": "flush_cadence_trace.jsonl",
            "expressiveness": "expressiveness_trace.jsonl",
            "narrative_expansion": "narrative_expansion_trace.jsonl",
            "semantic_structure": "semantic_structure_trace.jsonl",
            "frontend_burst": "frontend_burst_trace.jsonl",
            "visible_stream": "visible_stream_trace.jsonl",
            "verbosity": "verbosity_trace.jsonl",
            "conversation_naturalness": "conversation_naturalness_trace.jsonl",
            "human_alignment": "human_alignment_trace.jsonl",
        }
        
    def log(self, trace_name: str, data: dict):
        if trace_name not in self.traces:
            raise ValueError(f"Unknown trace {trace_name}")
            
        data["timestamp"] = time.time()
        file_path = self.trace_dir / self.traces[trace_name]
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
