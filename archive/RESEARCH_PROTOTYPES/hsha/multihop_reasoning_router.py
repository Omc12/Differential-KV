
from typing import List, Optional, Dict
from .symbolic_relationship_graph import SymbolicRelationshipGraph
from .traversal_legitimacy_guard import TraversalLegitimacyGuard

class MultihopReasoningRouter:
    """
    PHASE 21.5: MHSR - Chained Symbolic Traversal Engine.
    Routes recall through relationship chains.
    """
    def __init__(self, graph: SymbolicRelationshipGraph, guard: TraversalLegitimacyGuard):
        self.graph = graph
        self.guard = guard

    def route_multihop(self, start_object_id: str) -> List[str]:
        """
        Finds a legitimate traversal path starting from the given object.
        Returns a list of object IDs in the chain.
        """
        self.guard.reset()
        chain = [start_object_id]
        self.guard.enter_hop(start_object_id)
        
        curr = start_object_id
        while True:
            related = self.graph.get_related(curr)
            # Find the first legitimate related object
            next_obj = None
            for r_id in related:
                if self.guard.can_traverse(r_id):
                    next_obj = r_id
                    break
            
            if next_obj:
                chain.append(next_obj)
                self.guard.enter_hop(next_obj)
                curr = next_obj
            else:
                break
                
        return chain
