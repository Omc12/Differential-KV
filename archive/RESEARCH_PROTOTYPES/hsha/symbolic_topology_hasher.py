
import hashlib
from typing import List, Set

class SymbolicTopologyHasher:
    """
    PHASE 21.2: ISO - Structure-aware topology signatures.
    Hashes the delimiter skeleton of a symbolic object.
    """
    def __init__(self, delimiter_ids: Set[int]):
        self.delimiter_ids = delimiter_ids

    def hash_topology(self, tokens: List[int]) -> str:
        """
        Extracts the delimiter skeleton and produces a stable hash.
        Example: [10, 2, 20, 3, 30] where 2, 3 are delimiters -> skeleton [2, 3]
        """
        skeleton = [t for t in tokens if t in self.delimiter_ids]
        
        # If no delimiters, use a generic 'blob' signature
        if not skeleton:
            return "blob_" + hashlib.md5(b"none").hexdigest()[:8]
            
        skeleton_str = ",".join(map(str, skeleton))
        return hashlib.sha256(skeleton_str.encode()).hexdigest()

    def verify_integrity(self, tokens: List[int], expected_hash: str) -> bool:
        """Verifies if the token sequence matches the expected topology hash."""
        return self.hash_topology(tokens) == expected_hash
