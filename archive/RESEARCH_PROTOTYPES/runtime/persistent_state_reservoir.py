import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any

class PersistentStateReservoir:
    """
    PHASE 25: Persistent Cognitive State Reservoir
    Maintains tiny compressed persistent states (Working Memory Nuclei) across long horizons.
    """
    def __init__(self, d_model: int, compression_ratio: int = 16):
        self.d_model = d_model
        self.compression_ratio = compression_ratio
        self.reservoir_dim = max(1, d_model // compression_ratio)
        
        # Projections to/from the reservoir
        self.compressor = nn.Linear(d_model, self.reservoir_dim, bias=False)
        self.decompressor = nn.Linear(self.reservoir_dim, d_model, bias=False)
        
        # Initialize with semi-orthogonal weights to preserve signal
        nn.init.orthogonal_(self.compressor.weight)
        nn.init.orthogonal_(self.decompressor.weight)
        self.compressor.weight.data *= 0.5
        self.decompressor.weight.data *= 0.5
        
        self.states: Dict[str, torch.Tensor] = {}
        self.memory_nuclei: List[torch.Tensor] = []

    def store_state(self, key: str, state: torch.Tensor):
        """
        Compresses and stores a persistent state.
        """
        with torch.no_grad():
            compressed = self.compressor(state)
            self.states[key] = compressed.detach()

    def retrieve_state(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieves and decompresses a persistent state.
        """
        if key not in self.states:
            return None
        
        with torch.no_grad():
            return self.decompressor(self.states[key])

    def update_working_memory_nuclei(self, latent_states: torch.Tensor):
        """
        Maintains 'working memory nuclei' - the most persistent components of reasoning.
        """
        # latent_states shape: [seq_len, d_model] or [d_model]
        if latent_states.dim() == 1:
            latent_states = latent_states.unsqueeze(0)
            
        with torch.no_grad():
            # Project to nucleus space
            nuclei_candidates = self.compressor(latent_states)
            
            if not self.memory_nuclei:
                self.memory_nuclei = [nuclei_candidates.mean(dim=0)]
            else:
                # Running average of nuclei
                current_nucleus = self.memory_nuclei[-1]
                new_nucleus = 0.99 * current_nucleus + 0.01 * nuclei_candidates.mean(dim=0)
                self.memory_nuclei.append(new_nucleus)
                
                # Keep only recent history
                if len(self.memory_nuclei) > 100:
                    self.memory_nuclei.pop(0)

    def get_nucleus_injection(self) -> torch.Tensor:
        """
        Returns a signal from the working memory nuclei to stabilize current reasoning.
        """
        if not self.memory_nuclei:
            return torch.zeros(self.d_model)
            
        with torch.no_grad():
            return self.decompressor(self.memory_nuclei[-1])

    def get_overhead(self) -> float:
        """
        Calculates overhead of persistent state storage.
        """
        total_elements = sum(s.numel() for s in self.states.values())
        total_elements += sum(n.numel() for n in self.memory_nuclei)
        
        # Compare to full d_model states
        return (total_elements * 4) / (1024 * 1024) # in MB
