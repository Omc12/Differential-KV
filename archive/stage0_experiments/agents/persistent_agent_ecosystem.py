from typing import List, Dict, Any
from .distributed_cognition_exchange import DistributedCognitionExchange
from .shared_reasoning_substrate import SharedReasoningSubstrate

class PersistentAgentEcosystem:
    """
    Coordinates an ecosystem of persistent agents with shared cognition.
    Manages agent lifecycle, inheritance, and collective intelligence evolution.
    """
    def __init__(self):
        self.substrate = SharedReasoningSubstrate()
        self.exchange = DistributedCognitionExchange(self.substrate)
        self.agents = {}

    def register_agent(self, agent_id: str, capabilities: List[str]):
        """
        Registers a new agent in the ecosystem.
        """
        self.agents[agent_id] = {
            "capabilities": capabilities,
            "status": "active",
            "last_sync": None
        }
        print(f"Agent {agent_id} joined the ecosystem.")

    def perform_collective_sync(self):
        """
        Triggers a synchronization event across all active agents.
        """
        print("--- Performing Collective Cognitive Synchronization ---")
        for agent_id, info in self.agents.items():
            if info["status"] == "active":
                # In a real system, we would call agent.sync()
                # Here we just track the metadata
                info["last_sync"] = "now"
                
    def get_ecosystem_health(self) -> Dict[str, Any]:
        """
        Returns the overall health and diversity of the agent ecosystem.
        """
        return {
            "active_agents": len([a for a in self.agents.values() if a["status"] == "active"]),
            "substrate_stats": self.substrate.get_substrate_stats(),
            "diversity_index": 0.95 # Placeholder
        }
