import torch
import time
from runtime.multi_tier_kv_runtime import MultiTierKVRuntime
from runtime.adaptive_attention_density import AdaptiveAttentionDensity
from runtime.retrieval_aware_eviction import RetrievalAwareEviction
from runtime.context_entropy_scheduler import ContextEntropyScheduler
from runtime.sparse_anchor_graph import SparseAnchorGraph

class RetrievalStressSuite:
    """
    Stress tests the sparse runtime using long-context retrieval workloads.
    Measures retrieval retention vs sparsity level.
    """
    def __init__(self, context_length: int = 128000):
        self.context_length = context_length
        self.runtime = MultiTierKVRuntime(l1_capacity=16384, l2_capacity=128000)
        self.density_ctrl = AdaptiveAttentionDensity(target_density=0.4)
        self.eviction = RetrievalAwareEviction()
        self.scheduler = ContextEntropyScheduler(max_context=context_length)
        self.anchor_graph = SparseAnchorGraph()

    def run_needle_test(self, needle_pos: int):
        """
        Simulates a Needle-in-Haystack test.
        Places a 'needle' (unique token) at needle_pos and checks if it survives pruning.
        """
        print(f"Running Needle-in-Haystack at pos {needle_pos}...")
        
        # 1. Fill context with 'haystack' (random KV pairs)
        # We simulate this in chunks
        chunk_size = 1024
        for i in range(0, self.context_length, chunk_size):
            # Generate simulated KV
            k = torch.randn(1, 8, chunk_size, 64)
            v = torch.randn(1, 8, chunk_size, 64)
            
            # If this chunk contains the needle
            if i <= needle_pos < i + chunk_size:
                # Mark the needle token with high importance (simulated attention)
                # In real life, the model would attend to it.
                needle_idx = needle_pos - i
                k[:, :, needle_idx, :] *= 10 # Strong signal
            
            # Update runtime and pruning
            self.runtime.update_kv(0, k, v)
            
            # Simulate attention and update anchor graph
            # This would come from the forward pass
            dummy_attn = torch.randn(8, 1, k.size(-2)).softmax(dim=-1)
            self.anchor_graph.update_graph(dummy_attn)
            
        # 2. Check if needle is still in L1 or L2 cache
        # (Simplified: check if anchor graph identifies it)
        anchors = self.anchor_graph.get_anchors(top_k=2048)
        retrieved = needle_pos in anchors.tolist()
        
        return retrieved

    def benchmark_scaling(self):
        """Measures retrieval accuracy across different sparsity targets."""
        results = {}
        for density in [1.0, 0.5, 0.25, 0.1]:
            self.density_ctrl.target_density = density
            successes = 0
            for _ in range(5):
                pos = torch.randint(0, self.context_length, (1,)).item()
                if self.run_needle_test(pos):
                    successes += 1
            results[density] = successes / 5.0
        return results

if __name__ == "__main__":
    suite = RetrievalStressSuite(context_length=32768)
    results = suite.benchmark_scaling()
    print("Benchmark Results (Density vs Retrieval Accuracy):")
    print(results)
