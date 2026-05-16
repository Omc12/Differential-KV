from .cognitive_ir import CognitiveGraph, CIRNode, OpType

class MetalBackend:
    """Metal (Apple Silicon) backend for UCC."""
    
    def lower_graph(self, graph: CognitiveGraph):
        print(f"Lowering graph '{graph.name}' to Metal MPS graphs...")
        return {node_id: f"mps_graph_{node.op_type.value}" for node_id, node in graph.nodes.items()}
