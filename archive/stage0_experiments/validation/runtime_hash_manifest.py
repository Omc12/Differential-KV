import hashlib
import os
import json

class RuntimeHashManifest:
    """
    Generates hashes for all critical runtime components.
    Ensures that the code used for benchmarking is exactly what was reported.
    """
    def __init__(self, export_path: str = "results/reconstruction_18/runtime_hashes.json"):
        self.export_path = export_path
        self.hashes = {}

    def hash_file(self, filepath: str):
        if not os.path.exists(filepath):
            return None
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate_manifest(self, file_list: list):
        for filepath in file_list:
            h = self.hash_file(filepath)
            if h:
                self.hashes[os.path.basename(filepath)] = h
        
        with open(self.export_path, 'w') as f:
            json.dump(self.hashes, f, indent=4)
            
        return self.export_path
