from .cognitive_ir import CognitiveGraph, CIRNode, OpType

class ROCmBackend:
    """ROCm (AMD) backend for UCC."""
    
    def lower_graph(self, graph: CognitiveGraph):
        print(f"Lowering graph '{graph.name}' to ROCm/HIP kernels...")
        return {node_id: f"hip_kernel_{node.op_type.value}" for node_id, node in graph.nodes.items()}
