import torch

class AdaptiveChunkOverlap:
    """
    PHASE 18.5C: Bounded Overlap Scheduler.
    Maintains semantic continuity across chunk boundaries by re-processing 
    a small window of tokens from the previous chunk.
    """
    def __init__(self, overlap_size: int = 64):
        self.overlap = overlap_size

    def get_chunks(self, input_ids, chunk_size):
        """
        Generates overlapping chunks from input_ids.
        Returns (chunk, stride_start, stride_len)
        """
        seq_len = input_ids.shape[1]
        chunks = []
        
        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            chunk = input_ids[:, start:end]
            
            # stride_len is the number of NEW tokens in this chunk
            if start == 0:
                stride_start = 0
                stride_len = end
            else:
                stride_start = self.overlap
                stride_len = end - start - self.overlap
                
            chunks.append((chunk, stride_start, stride_len))
            
            if end == seq_len:
                break
            # Advance start by chunk_size minus overlap
            start += (chunk_size - self.overlap)
            
        return chunks
