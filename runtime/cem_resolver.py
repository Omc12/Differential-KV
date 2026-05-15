
import torch
from typing import Optional, List, Dict, Any, Tuple
from runtime.crs_resolver import CRSResolver
from cem.cognitive_market_coordinator import CognitiveMarketCoordinator
from cem.symbolic_value_bidding_engine import SymbolicValueBiddingEngine
from cem.cooperative_execution_exchange import CooperativeExecutionExchange
from cem.adaptive_resource_auctioneer import AdaptiveResourceAuctioneer
from cem.market_integrity_guard import MarketIntegrityGuard

class CEMResolver(CRSResolver):
    """
    PHASE 23.5: CEM (Cognitive Execution Markets).
    Implements decentralized cognition execution economies.
    Architectural Shift: Decentralized Cognition Execution Economies.
    """
    def __init__(self, tokenizer, anchor_budget: int = 6144, fidelity_budget: int = 1024):
        super().__init__(tokenizer, anchor_budget, fidelity_budget)
        
        config = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        
        # CEM Components
        self.market_coordinator = CognitiveMarketCoordinator(config)
        self.bidding_engine = SymbolicValueBiddingEngine(config)
        self.cooperative_exchange = CooperativeExecutionExchange(config)
        self.resource_auctioneer = AdaptiveResourceAuctioneer(config)
        self.market_guard = MarketIntegrityGuard(config)
        
        # CEM Metrics
        self.cem_metrics = {
            "market_allocation_efficiency": 1.0,
            "symbolic_bidding_stability": 1.0,
            "cooperative_exchange_health": 1.0,
            "anti_monopoly_integrity": 1.0,
            "symbolic_continuity": 1.0,
            "market_stability": 1.0
        }

    def resolve_and_prune(self, past_key_values, hidden_states, chunk_input_ids, attention_probs=None):
        """
        CEM-aware Pruning & Decentralized Negotiation.
        Execution regions negotiate residency through cognitive markets.
        """
        # 1. Base CRS logic (which includes ARC, PER, ELF, KRX, ESM, AEG, SRE)
        pruned_pkv, indices = super().resolve_and_prune(past_key_values, hidden_states, chunk_input_ids, attention_probs)
        
        # 2. CEM: Decentralized Negotiation
        seq_len = hidden_states.shape[1]
        
        # Identify candidate blocks for bidding
        block_size = 128
        num_blocks = (seq_len + block_size - 1) // block_size
        bids = []
        
        for i in range(num_blocks):
            # Generate bid
            hub_id = self.current_hub_id if hasattr(self, 'current_hub_id') else None
            importance = self.crs_importance_estimator.estimate_importance(hub_id, 1.0, 5)
            forecast = self.activation_forecaster.forecast_activation(i, self.current_step)
            
            bid = self.bidding_engine.generate_bid(i, importance, forecast)
            bids.append((i, bid))
            
        # 3. Market Coordination & Auction
        available_slots = 16 # Mock capacity
        market_winners = self.resource_auctioneer.run_auction(bids, available_slots)
        
        # 4. Cooperative Coalition Formation
        if hub_id and len(market_winners) > 0:
            self.cooperative_exchange.form_coalition(hub_id, market_winners[:4])
            
        # 5. Market Integrity Guard
        self.market_guard.validate_market_step(market_winners)
        
        # Update metrics
        self._update_cem_metrics()
        
        return pruned_pkv, indices

    def _update_cem_metrics(self):
        """Aggregates metrics from CEM components."""
        c_m = self.market_coordinator.get_metrics()
        b_m = self.bidding_engine.get_metrics()
        e_m = self.cooperative_exchange.get_metrics()
        a_m = self.resource_auctioneer.get_metrics()
        g_m = self.market_guard.get_metrics()
        
        self.cem_metrics["market_allocation_efficiency"] = c_m["market_allocation_efficiency"]
        self.cem_metrics["symbolic_bidding_stability"] = b_m["symbolic_bidding_stability"]
        self.cem_metrics["cooperative_exchange_health"] = e_m["cooperative_exchange_health"]
        self.cem_metrics["anti_monopoly_integrity"] = g_m["anti_monopoly_integrity"]
        self.cem_metrics["symbolic_continuity"] = g_m["symbolic_continuity"]
        self.cem_metrics["market_stability"] = c_m["market_stability"]

    def get_cem_stats(self) -> Dict[str, Any]:
        """Returns summarized metrics for Phase 23.5 validation."""
        self._update_cem_metrics()
        return self.cem_metrics
