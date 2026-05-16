from typing import List, Set, Dict

class BenchmarkComponentRegistry:
    """
    Tracks exactly which systems participated in a benchmark run.
    """
    ALL_COMPONENTS = {
        "embeddings",
        "tokenizer",
        "logits",
        "mlp",
        "triton_kernels",
        "sparse_attention",
        "kv_virtualization",
        "hf_dispatch",
        "cuda_graphs",
        "sampling",
        "streaming",
        "batching",
        "concurrency",
        "serving_overhead"
    }

    def __init__(self):
        self.participating: Set[str] = set()

    def register(self, component: str):
        if component not in self.ALL_COMPONENTS:
            print(f"[BIC] WARNING: Registering unknown component: {component}")
        self.participating.add(component)

    def get_participation_manifest(self) -> List[str]:
        return sorted(list(self.participating))

    def get_excluded_manifest(self) -> List[str]:
        return sorted(list(self.ALL_COMPONENTS - self.participating))

# Global singleton
registry = BenchmarkComponentRegistry()
