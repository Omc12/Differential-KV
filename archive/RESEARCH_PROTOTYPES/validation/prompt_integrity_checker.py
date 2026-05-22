import hashlib
import json
import os

class PromptIntegrityChecker:
    """
    PHASE 18.1C: Ensures prompt integrity for deterministic benchmarking.
    """
    def __init__(self, export_path: str = "results/reconstruction_18_1/raw_prompt_hashes.json"):
        self.export_path = export_path
        self.hashes = {}

    def hash_prompt(self, name: str, text: str):
        p_hash = hashlib.sha256(text.encode()).hexdigest()
        self.hashes[name] = p_hash
        self.export()
        return p_hash

    def export(self):
        with open(self.export_path, 'w') as f:
            json.dump(self.hashes, f, indent=4)
