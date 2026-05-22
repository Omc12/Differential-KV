import torch
from typing import Dict, List

class RecursiveAttractorSummarization:
    """
    Summarizes stable attractors into persistent reasoning motifs.
    Used for long-horizon cognition rollups.
    """
    def __init__(self):
        self.persistent_motifs = []
        
    def summarize_attractors(self, active_attractors: Dict) -> List[torch.Tensor]:
        """
        Extracts stable features from the current attractor ecosystem.
        """
        summaries = []
        for aid, meta in active_attractors.items():
            if meta['health'] > 1.2: # Only summarize very stable attractors
                # In a real implementation, we'd use a decoder or LLM call to 'name' the attractor
                # Here we use the latent center as the summary
                summaries.append(torch.tensor(meta.get('center', [0.0])))
                
        self.persistent_motifs.extend(summaries)
        # Limit persistence to prevent memory bloat
        if len(self.persistent_motifs) > 100:
            self.persistent_motifs = self.persistent_motifs[-100:]
            
        return summaries
        
    def get_motifs(self) -> List[torch.Tensor]:
        return self.persistent_motifs
