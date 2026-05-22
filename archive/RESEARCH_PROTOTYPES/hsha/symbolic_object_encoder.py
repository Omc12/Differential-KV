
import hashlib
from typing import List, Set, Dict

class SymbolicObjectEncoder:
    """
    PHASE 21.0: Extracts structural fingerprints and topologies from token sequences.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.delimiter_chars = {"-", "_", ":", "/", ".", "=", "{", "}", "[", "]", "(", ")", ",", ";", "@", "#", "$", "%", "^", "*", "+", "|", "<", ">"}
        self.delimiter_ids = self._initialize_delimiters()

    def _initialize_delimiters(self) -> Set[int]:
        d_ids = set()
        for char in self.delimiter_chars:
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            d_ids.update(ids)
        return d_ids

    def extract_topology(self, tokens: List[int]) -> str:
        """
        Extracts the 'delimiter skeleton' of a sequence.
        Example: 'abc-123:xyz' -> '-:' (hashed)
        """
        skeleton = []
        for t in tokens:
            if t in self.delimiter_ids:
                skeleton.append(str(t))
        
        skeleton_str = ",".join(skeleton)
        return hashlib.sha256(skeleton_str.encode()).hexdigest()[:12]

    def get_structural_fingerprint(self, tokens: List[int]) -> str:
        """
        Creates a fingerprint based on delimiter positions and token counts between them.
        """
        fingerprint = []
        count = 0
        for t in tokens:
            if t in self.delimiter_ids:
                fingerprint.append(f"d{t}c{count}")
                count = 0
            else:
                count += 1
        fingerprint.append(f"e{count}")
        
        fp_str = "-".join(fingerprint)
        return hashlib.sha256(fp_str.encode()).hexdigest()[:16]

    def serialize(self, tokens: List[int]) -> Dict:
        """Serializes a symbolic object for hub storage."""
        return {
            "tokens": tokens,
            "topology_hash": self.extract_topology(tokens),
            "fingerprint": self.get_structural_fingerprint(tokens)
        }
