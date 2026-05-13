import torch
from revival.bounded_summary_memory import BoundedSummaryMemory

class ContextRollupEngine:
    """
    Manages the rollup of context chunks into bounded summaries.
    """

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size
        self.memory = BoundedSummaryMemory()

    def process_sequence(self, full_sequence):
        """
        Processes a long sequence by rolling up chunks into summaries.
        """
        seq_len = len(full_sequence)
        summarized_sequence = []
        
        for i in range(0, seq_len, self.chunk_size):
            chunk = full_sequence[i:i+self.chunk_size]
            if len(chunk) == self.chunk_size:
                summary = self.memory.create_summary_token(chunk)
                summarized_sequence.extend(summary)
            else:
                # Last chunk kept as is (sliding window behavior)
                summarized_sequence.extend(chunk)
                
        return summarized_sequence

if __name__ == "__main__":
    engine = ContextRollupEngine(chunk_size=100)
    full_seq = list(range(500))
    rolled_up = engine.process_sequence(full_seq)
    print(f"Original sequence: {len(full_seq)}, Rolled up: {len(rolled_up)}")
