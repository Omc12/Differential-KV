import enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import torch

class OpType(enum.Enum):
    ATTRACTOR_MAP = "attractor_map"
    GEOMETRIC_ATTENTION = "geometric_attention"
    RESONANCE_SCHEDULE = "resonance_schedule"
    MANIFOLD_PROJECTION = "manifold_projection"
    STABILIZATION_FLOW = "stabilization_flow"
    ANCHOR_RESTORATION = "anchor_restoration"
    COGNITIVE_ROUTING = "cognitive_routing"

@dataclass
class CIRNode:
    id: str
    op_type: OpType
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CognitiveGraph:
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, CIRNode] = {}
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        
    def add_node(self, node: CIRNode):
        self.nodes[node.id] = node
        
    def to_dict(self):
        return {
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": {k: vars(v) for k, v in self.nodes.items()}
        }

@dataclass
class GeometricTensor:
    """A tensor with geometric metadata for manifold awareness."""
    data: torch.Tensor
    manifold_id: str
    curvature: Optional[torch.Tensor] = None
    drift_history: List[float] = field(default_factory=list)
