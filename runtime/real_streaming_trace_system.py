import json
import time
from pathlib import Path
from typing import Dict, Any

class RealStreamingTraceSystem:
    """
    Stage 4B.1.5 RTA: Real Streaming Trace System.
    Persists exactly the 10 reality-verified trace profiles.
    """
    def __init__(self, trace_dir: Path):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        
        # 10 designated physical JSONL trace files
        self.files = {
            "emitted_token": open(self.trace_dir / "emitted_token_trace.jsonl", "w", encoding="utf-8"),
            "wallclock": open(self.trace_dir / "wallclock_trace.jsonl", "w", encoding="utf-8"),
            "real_generation": open(self.trace_dir / "real_generation_trace.jsonl", "w", encoding="utf-8"),
            "ttft": open(self.trace_dir / "ttft_trace.jsonl", "w", encoding="utf-8"),
            "intertoken": open(self.trace_dir / "intertoken_trace.jsonl", "w", encoding="utf-8"),
            "stream_completion": open(self.trace_dir / "stream_completion_trace.jsonl", "w", encoding="utf-8"),
            "throughput_truth": open(self.trace_dir / "throughput_truth_trace.jsonl", "w", encoding="utf-8"),
            "ollama_comparison": open(self.trace_dir / "ollama_comparison_trace.jsonl", "w", encoding="utf-8"),
            "replay_vs_real": open(self.trace_dir / "replay_vs_real_trace.jsonl", "w", encoding="utf-8"),
            "scheduler_vs_real": open(self.trace_dir / "scheduler_vs_real_trace.jsonl", "w", encoding="utf-8"),
        }

    def _write_record(self, trace_key: str, data: Dict[str, Any]):
        if trace_key in self.files:
            record = {"timestamp": time.time(), **data}
            f = self.files[trace_key]
            f.write(json.dumps(record) + "\n")
            f.flush()

    def record_emitted_token(self, index: int, token_text: str, latency: float):
        self._write_record("emitted_token", {"token_idx": index, "token": token_text, "latency": latency})

    def record_wallclock(self, duration: float, label: str):
        self._write_record("wallclock", {"duration_sec": duration, "label": label})

    def record_real_generation(self, prompt: str, output: str, tokens: int, duration: float):
        self._write_record("real_generation", {
            "prompt_len": len(prompt),
            "output_len": len(output),
            "tokens_generated": tokens,
            "duration_sec": duration,
            "real_tps": tokens / max(0.01, duration)
        })

    def record_ttft(self, ttft_ms: float):
        self._write_record("ttft", {"ttft_ms": ttft_ms})

    def record_intertoken(self, index: int, latency_ms: float):
        self._write_record("intertoken", {"step": index, "latency_ms": latency_ms})

    def record_stream_completion(self, duration: float, success: bool):
        self._write_record("stream_completion", {"duration_sec": duration, "success": success})

    def record_throughput_truth(self, real_tps: float, total_tokens: int):
        self._write_record("throughput_truth", {"real_tps": real_tps, "total_tokens": total_tokens})

    def record_ollama_comparison(self, diffkv_tps: float, ollama_tps: float):
        self._write_record("ollama_comparison", {"diffkv_tps": diffkv_tps, "ollama_tps": ollama_tps})

    def record_replay_vs_real(self, replay_tps: float, real_tps: float):
        self._write_record("replay_vs_real", {"replay_tps": replay_tps, "real_tps": real_tps})

    def record_scheduler_vs_real(self, scheduler_tps: float, real_tps: float):
        self._write_record("scheduler_vs_real", {"scheduler_tps": scheduler_tps, "real_tps": real_tps})

    def close(self):
        for f in self.files.values():
            try:
                f.close()
            except:
                pass
