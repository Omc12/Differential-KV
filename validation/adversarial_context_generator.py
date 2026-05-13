"""
validation/adversarial_context_generator.py

Phase 12.5A: Adversarial Context Generator
Generates noisy, confusing, or misleading context to test the robustness
of semantic retrieval.
"""

import random
from typing import List

class AdversarialContextGenerator:
    """
    Creates "distractor" text designed to confuse semantic similarity metrics.
    """
    
    def generate_distractors(self, original_text: str, count: int = 5) -> List[str]:
        """
        Creates variations of the original text that use similar words but
        have different semantic meanings (e.g., negations).
        """
        distractors = []
        words = original_text.split()
        
        for i in range(count):
            if len(words) < 3:
                distractors.append(original_text + f" [distractor_{i}]")
                continue
                
            # Randomly swap words
            idx1, idx2 = random.sample(range(len(words)), 2)
            new_words = list(words)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
            # Add negation
            if " is " in original_text:
                distractors.append(" ".join(new_words).replace(" is ", " is NOT "))
            else:
                distractors.append("NOT " + " ".join(new_words))
                
        return distractors

    def inject_noise(self, text: str, noise_level: float = 0.1) -> str:
        """Randomly replaces characters with noise."""
        chars = list(text)
        num_noisy = int(len(chars) * noise_level)
        indices = random.sample(range(len(chars)), num_noisy)
        
        for idx in indices:
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz1234567890!@#$%")
            
        return "".join(chars)
