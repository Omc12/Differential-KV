"""
analysis/collapse_region_catalog.py

A database and catalog of known cognitive collapse regions for various model architectures.
Provides scientific transparency on where Differential KV fails.
"""

import json
import os

class CollapseRegionCatalog:
    def __init__(self, catalog_path: str = "results/phase38/collapse_catalog.json"):
        self.catalog_path = catalog_path
        self.catalog = self._load()

    def _load(self):
        if os.path.exists(self.catalog_path):
            with open(self.catalog_path, "r") as f:
                return json.load(f)
        return {
            "architectures": {
                "llama": [],
                "qwen": [],
                "mistral": []
            },
            "global_patterns": []
        }

    def add_failure_mode(self, arch: str, context_len: int, sparsity: float, trigger: str, symptoms: str):
        entry = {
            "context_len": context_len,
            "sparsity": sparsity,
            "trigger": trigger,
            "symptoms": symptoms,
            "timestamp": os.path.getmtime(self.catalog_path) if os.path.exists(self.catalog_path) else 0
        }
        if arch in self.catalog["architectures"]:
            self.catalog["architectures"][arch].append(entry)
        else:
            self.catalog["architectures"][arch] = [entry]
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.catalog_path), exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(self.catalog, f, indent=4)

if __name__ == "__main__":
    catalog = CollapseRegionCatalog()
    catalog.add_failure_mode(
        arch="qwen",
        context_len=131072,
        sparsity=0.01,
        trigger="Extreme sparse pressure",
        symptoms="Logit explosion and repetitive token generation"
    )
    print("Catalog updated.")
