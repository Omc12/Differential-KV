import networkx as nx
import matplotlib.pyplot as plt
from compiler.cognitive_ir import CognitiveGraph

class GraphIRVisualizer:
    """Visualizes Cognitive IR graphs."""
    
    @staticmethod
    def plot_graph(graph: CognitiveGraph, output_path: str):
        G = nx.DiGraph()
        
        for node_id, node in graph.nodes.items():
            G.add_node(node_id, label=node.op_type.value)
            for inp in node.inputs:
                G.add_edge(inp, node_id)
            for outp in node.outputs:
                G.add_edge(node_id, outp)
                
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold")
        plt.title(f"Cognitive Graph IR: {graph.name}")
        plt.savefig(output_path)
        print(f"Graph visualization saved to {output_path}")
