"""
economy/manifold_exchange_market.py

Market-based mechanism for manifold reuse and propagation.
"""

import torch
from typing import Dict, List, Optional, Any

class ManifoldExchangeMarket:
    """
    Facilitates the 'buying' and 'selling' (sharing) of cognitive manifolds.
    Uses 'cognitive credits' (a mock metric) to prioritize resource allocation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_credits = {} # agent_id -> credits
        self.market_prices = {} # manifold_id -> price

    def register_participation(self, agent_id: str):
        """Adds an agent to the market."""
        if agent_id not in self.agent_credits:
            self.agent_credits[agent_id] = 100.0

    def propose_exchange(self, provider_id: str, requester_id: str, manifold_id: str) -> bool:
        """
        Executes a manifold exchange between two agents.
        """
        price = self.market_prices.get(manifold_id, 10.0)
        
        if self.agent_credits.get(requester_id, 0) >= price:
            self.agent_credits[requester_id] -= price
            self.agent_credits[provider_id] = self.agent_credits.get(provider_id, 0) + price
            return True
        return False

    def update_prices(self, demand_metrics: Dict[str, int]):
        """Adjusts market prices based on manifold demand."""
        for mid, demand in demand_metrics.items():
            current_price = self.market_prices.get(mid, 10.0)
            # Increase price if demand is high
            self.market_prices[mid] = current_price * (1.0 + 0.1 * demand)
            # Limit price
            self.market_prices[mid] = min(1000.0, self.market_prices[mid])

    def get_market_state(self) -> Dict[str, Any]:
        """Returns the current state of the cognitive economy."""
        return {
            "total_credits": sum(self.agent_credits.values()),
            "active_exchanges": len(self.market_prices),
            "average_price": sum(self.market_prices.values()) / max(1, len(self.market_prices))
        }
