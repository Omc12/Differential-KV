"""
validation/semantic_decoy_generator.py

Phase 12.5D: Semantic Decoy Generator
Systematically generates adversarial decoys specifically targeted at
weaknesses in embedding-based sparse retrieval.
"""

from typing import List

class SemanticDecoyGenerator:
    """
    Generates decoys that have high lexical overlap but contradictory semantics.
    """
    
    def generate_contradictions(self, fact: str) -> List[str]:
        """e.g. 'The API is thread-safe' -> 'The API is NOT thread-safe'"""
        decoys = []
        words = fact.split()
        
        # Simple negations
        if " is " in fact:
            decoys.append(fact.replace(" is ", " is not "))
        elif " has " in fact:
            decoys.append(fact.replace(" has ", " lacks "))
            
        # Antonyms (hardcoded mock for demonstration)
        antonyms = {"enable": "disable", "start": "stop", "true": "false", "fast": "slow", "uses": "ignores", "persistent": "ephemeral"}
        for word in words:
            clean_word = word.strip(".,!").lower()
            if clean_word in antonyms:
                decoys.append(fact.replace(word, antonyms[clean_word]))
                
        return decoys
