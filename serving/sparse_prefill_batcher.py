"""
Sparse Prefill Batcher.
Batches long-context prefill phases without stalling active decode batches.
"""

class SparsePrefillBatcher:
    def process_prefill(self, reqs):
        return sum([r.get('ctx', 0) for r in reqs])
