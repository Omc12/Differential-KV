import torch

class CUDAGraphDecodeExecutor:
    """
    PHASE 11B: REAL GPU EXECUTION OPTIMIZATION
    
    Uses CUDA Graphs to capture and replay the decode sequence.
    Reduces the overhead of launching multiple kernels per token.
    """
    def __init__(self, model, max_seq_len: int = 1024):
        self.model = model
        self.max_seq_len = max_seq_len
        self.graph = None
        self.static_input_ids = None
        self.static_outputs = None

    def capture_graph(self, sample_input_ids):
        """
        Captures the model's execution trace into a CUDA Graph.
        """
        self.static_input_ids = sample_input_ids.clone()
        
        # Warmup
        for _ in range(3):
            self.model(self.static_input_ids)
            
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_outputs = self.model(self.static_input_ids)

    def replay_step(self, input_ids):
        """
        Executes a decode step using the captured graph.
        """
        if self.graph is None:
            self.capture_graph(input_ids)
            
        self.static_input_ids.copy_(input_ids)
        self.graph.replay()
        return self.static_outputs
