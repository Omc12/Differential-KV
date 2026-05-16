import json
import os
import time

class TokenTraceArchiver:
    """
    Archives every token generated during a benchmark run for audit.
    MANDATORY for verifying TPS and output quality.
    """
    def __init__(self, run_id: str, export_dir: str = "results/reconstruction_18/"):
        self.run_id = run_id
        self.export_dir = export_dir
        self.traces = []

    def record_trace(self, prompt, generated_text, tokens, elapsed_time):
        trace = {
            "timestamp": time.time(),
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens": len(tokens),
            "elapsed_time": elapsed_time,
            "tps": len(tokens) / elapsed_time if elapsed_time > 0 else 0
        }
        self.traces.append(trace)

    def archive(self):
        filename = f"token_trace_{self.run_id}.jsonl"
        path = os.path.join(self.export_dir, filename)
        
        with open(path, 'w') as f:
            for trace in self.traces:
                f.write(json.dumps(trace) + "\n")
                
        return path
