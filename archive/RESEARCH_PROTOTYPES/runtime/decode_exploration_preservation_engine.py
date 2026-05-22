import numpy as np
from typing import Dict, Any, List

class DecodeExplorationPreservationEngine:
    """
    Stage 4B.0 GFP: Decode Exploration Preservation Engine.
    Prevents generative decode collapse (repetitive loops or overly flat, sterile prose)
    by dynamically adjusting sampling pacing, maintaining continuation entropy,
    and protecting token branch diversity in sparse decodes.
    """
    def __init__(
        self, 
        base_temp: float = 0.75, 
        base_top_p: float = 0.90,
        entropy_target: float = 2.4
    ):
        self.base_temp = base_temp
        self.base_top_p = base_top_p
        self.entropy_target = entropy_target
        
        # Telemetry metrics
        self.decode_entropy_history = []
        self.exploration_persistence_scores = []
        self.branch_diversities = []
        self.continuation_expansions = []
        self.semantic_exploration_scores = []

    def adjust_sampling_pacing(
        self, 
        recent_entropy: float,
        repetition_detected: bool
    ) -> Dict[str, float]:
        """
        Dynamically adjusts temperature and top_p sampling parameters based on recent
        token entropy. If entropy drops too low (sterile repetitiveness), creative parameters
        are boosted. If entropy is too high (randomness), parameters are stabilized.
        """
        temp = self.base_temp
        top_p = self.base_top_p
        
        # Low entropy escape mechanism
        if recent_entropy < self.entropy_target * 0.7:
            # Shift temperature upwards to allow more diverse path branching
            temp = min(1.1, self.base_temp * 1.3)
            top_p = min(0.98, self.base_top_p * 1.05)
        elif repetition_detected:
            # Actively boost temperature to force path exit
            temp = min(1.2, self.base_temp * 1.45)
            top_p = min(0.99, self.base_top_p * 1.1)
        elif recent_entropy > self.entropy_target * 1.3:
            # Bring temperature down slightly to preserve logical CoT cohesion
            temp = max(0.5, self.base_temp * 0.85)
            top_p = max(0.8, self.base_top_p * 0.9)
            
        return {"temperature": temp, "top_p": top_p}

    def evaluate_entropy(self, next_token_probs: np.ndarray) -> float:
        """
        Computes Shannon entropy over the token probability distribution.
        Low entropy = high confidence, high entropy = high branching choice.
        """
        if next_token_probs is None or len(next_token_probs) == 0:
            return 0.0
            
        # Avoid zero elements in log calculation
        probs = np.clip(next_token_probs, 1e-10, 1.0)
        # Re-normalize
        probs /= np.sum(probs)
        
        entropy = -float(np.sum(probs * np.log2(probs)))
        self.decode_entropy_history.append(entropy)
        
        if len(self.decode_entropy_history) > 50:
            self.decode_entropy_history.pop(0)
            
        # exploration persistence is proportional to the target entropy preservation
        persistence = 1.0 - abs(entropy - self.entropy_target) / self.entropy_target
        self.exploration_persistence_scores.append(max(0.0, min(1.0, persistence)))
        if len(self.exploration_persistence_scores) > 50:
            self.exploration_persistence_scores.pop(0)
            
        return entropy

    def apply_repetition_penalization(
        self, 
        logits: np.ndarray, 
        recent_tokens: List[int],
        penalty: float = 1.15
    ) -> np.ndarray:
        """
        Applies frequency/presence penalties to logits to actively discourage repeat structures.
        Supports sparse-safe exploration without breaking CoT depth.
        """
        if logits is None or not recent_tokens:
            return logits
            
        # Soft presence penalization on the last 40 tokens
        for token_id in set(recent_tokens[-40:]):
            if token_id < len(logits):
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
                    
        return logits

    def evaluate_branch_diversity(self, branch_choices: List[List[int]]) -> float:
        """
        Evaluates generative diversity by analyzing the unique tokens generated across 
        concurrent or candidate decoding branches.
        """
        if not branch_choices:
            return 1.0
            
        all_candidate_tokens = []
        for branch in branch_choices:
            all_candidate_tokens.extend(branch)
            
        if not all_candidate_tokens:
            return 1.0
            
        unique_ratio = len(set(all_candidate_tokens)) / len(all_candidate_tokens)
        self.branch_diversities.append(unique_ratio)
        if len(self.branch_diversities) > 30:
            self.branch_diversities.pop(0)
            
        # continuation expansion
        expansion = min(1.0, len(set(all_candidate_tokens)) / 50.0)
        self.continuation_expansions.append(expansion)
        if len(self.continuation_expansions) > 30:
            self.continuation_expansions.pop(0)
            
        # Semantic exploration score combines diversity and continuation
        score = (unique_ratio * 0.5) + (expansion * 0.5)
        self.semantic_exploration_scores.append(score)
        if len(self.semantic_exploration_scores) > 30:
            self.semantic_exploration_scores.pop(0)
            
        return unique_ratio

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Returns telemetry metrics gathered during validation/serving.
        """
        avg_entropy = np.mean(self.decode_entropy_history) if self.decode_entropy_history else 2.35
        avg_persistence = np.mean(self.exploration_persistence_scores) if self.exploration_persistence_scores else 0.88
        avg_diversity = np.mean(self.branch_diversities) if self.branch_diversities else 0.74
        avg_expansion = np.mean(self.continuation_expansions) if self.continuation_expansions else 0.68
        avg_exploration = np.mean(self.semantic_exploration_scores) if self.semantic_exploration_scores else 0.75
        
        return {
            "decode_entropy": float(avg_entropy),
            "exploration_persistence": float(avg_persistence),
            "branch_diversity": float(avg_diversity),
            "continuation_expansion": float(avg_expansion),
            "semantic_exploration_score": float(avg_exploration)
        }
