import json
import hashlib
from dataclasses import dataclass, asdict
import torch

@dataclass
class BenchmarkLock:
    model_name: str
    quantization: str
    context_length: int
    generation_length: int
    concurrency: int
    gpu_model: str
    vram_usage_gb: float
    prefill_decode_separation: bool
    retrieval_density: float

class MethodologyLock:
    def __init__(self, lock_file="results/methodology_lock.json"):
        self.lock_file = lock_file

    def generate_lock(self, params: BenchmarkLock) -> str:
        data = asdict(params)
        lock_str = json.dumps(data, sort_keys=True)
        lock_hash = hashlib.sha256(lock_str.encode()).hexdigest()
        return lock_hash

    def enforce_lock(self, params: BenchmarkLock, expected_hash: str):
        actual_hash = self.generate_lock(params)
        if actual_hash != expected_hash:
            raise ValueError(f"Methodology mismatch! Expected hash {expected_hash}, got {actual_hash}. Re-run with exact configuration.")

    def log_methodology(self, params: BenchmarkLock):
        print(f"--- Benchmark Methodology Lock ---")
        for k, v in asdict(params).items():
            print(f"{k}: {v}")
        print(f"Hash: {self.generate_lock(params)}")
