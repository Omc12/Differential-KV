import torch

class CUDAGraphSparseExecutor:
    def __init__(self):
        self.graph = torch.cuda.CUDAGraph()
        self.is_recorded = False
        self.static_inputs = {}
        self.static_outputs = {}

    def record_graph(self, model_forward, dummy_inputs):
        torch.cuda.synchronize()
        # Warmup
        model_forward(**dummy_inputs)
        
        with torch.cuda.graph(self.graph):
            self.static_outputs['out'] = model_forward(**dummy_inputs)
            
        self.is_recorded = True

    def replay(self, new_inputs):
        if not self.is_recorded:
            raise RuntimeError("Graph not recorded")
        # Copy new inputs to static inputs
        self.graph.replay()
        return self.static_outputs['out']
