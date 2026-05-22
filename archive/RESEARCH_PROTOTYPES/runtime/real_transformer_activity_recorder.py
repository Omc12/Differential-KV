import os
import time
import json
from pathlib import Path

class RealTransformerActivityRecorder:
    """
    RHD Phase 41.4.6 — Real Transformer Activity Recorder.
    Tracks real forward passes, layer invocations, attention calls, decode tokens,
    and sequence lengths. Must derive from real model execution paths.
    Persists raw activity to JSONL.
    """
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.trace_dir = self.workspace_root / "traces/stage3b/phase_41_4_6_rhd"
        self.trace_path = self.trace_dir / "raw_transformer_activity_trace.jsonl"
        
        self.forward_passes = 0

    def record_forward_pass(self, step: int, seq_len: int, num_layers: int):
        self.forward_passes += 1
        self._write_activity({
            "timestamp": time.time(),
            "activity_type": "forward_pass",
            "forward_pass_id": self.forward_passes,
            "step": step,
            "seq_len": seq_len,
            "num_layers": num_layers
        })

    def record_layer_invocation(self, step: int, layer_idx: int, seq_len: int, mode: str = "sparse"):
        self._write_activity({
            "timestamp": time.time(),
            "activity_type": "layer_invocation",
            "forward_pass_id": self.forward_passes,
            "step": step,
            "layer_idx": layer_idx,
            "seq_len": seq_len,
            "execution_mode": mode
        })

    def record_attention_call(self, step: int, layer_idx: int, q_shape: list, k_shape: list, is_sparse: bool):
        self._write_activity({
            "timestamp": time.time(),
            "activity_type": "attention_call",
            "forward_pass_id": self.forward_passes,
            "step": step,
            "layer_idx": layer_idx,
            "query_shape": q_shape,
            "key_shape": k_shape,
            "is_sparse": is_sparse
        })

    def record_decode_token(self, step: int, token_id: int, decoded_text: str):
        self._write_activity({
            "timestamp": time.time(),
            "activity_type": "decode_token",
            "step": step,
            "token_id": token_id,
            "text": decoded_text
        })

    def _write_activity(self, data: dict):
        os.makedirs(self.trace_dir, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
