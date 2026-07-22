import json
import os

class PublicRuntimeMatrix:
    """
    Phase 18C: Manages the comparison matrix between DKV and public runtimes.
    Ensures identical checkpoint/quantization for fair comparison.
    """
    def __init__(self, export_dir: str = "results/reconstruction_18/"):
        self.export_dir = export_dir
        self.matrix = {
            "baselines": ["vLLM", "llama.cpp", "TensorRT-LLM"],
            "target": "Differential KV",
            "metrics": ["TPS", "TTFT", "VRAM (GB)", "VRAM Efficiency"],
            "runs": []
        }

    def add_run(self, runtime, config, metrics):
        self.matrix["runs"].append({
            "runtime": runtime,
            "config": config,
            "metrics": metrics
        })
        self.export()

    def export(self):
        path = os.path.join(self.export_dir, "public_runtime_matrix.json")
        with open(path, 'w') as f:
            json.dump(self.matrix, f, indent=4)
        return path

if __name__ == "__main__":
    matrix = PublicRuntimeMatrix()
    print("Public Runtime Matrix initialized.")
