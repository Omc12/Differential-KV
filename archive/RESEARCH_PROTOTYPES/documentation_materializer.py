"""
documentation_materializer.py

Documentation generator for Differential KV.
Produces quickstart, API references, and deployment guides.
"""

import os
from typing import Dict, Any

class DocumentationMaterializer:
    """
    Automates document generation for developers.
    """
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def materialize_all(self):
        """Generates the full documentation set."""
        self._generate_quickstart()
        self._generate_api_reference()
        self._generate_deployment_guide()

    def _generate_quickstart(self):
        content = """# Differential KV Quickstart

## 1. Installation
```bash
pip install -e .[gpu]
```

## 2. Diagnostics
Check your environment:
```bash
dkv doctor
```

## 3. Launch Server
```bash
dkv serve --model Qwen/Qwen2.5-7B-Instruct
```

## 4. Run Examples
Check the `examples/` directory for integrations with LangChain, LlamaIndex, and more.
"""
        with open(os.path.join(self.output_dir, "QUICKSTART.md"), "w") as f:
            f.write(content)

    def _generate_api_reference(self):
        content = """# API Reference

## CLI
- `dkv serve`: Launch inference gateway.
- `dkv benchmark`: Run OBS suite.
- `dkv doctor`: Run environment checks.

## Python Adapters
- `DKVHFAdapter`: HuggingFace compatibility.
- `DKVSparseLLM`: LangChain integration.
- `DKVLlamaIndexAdapter`: LlamaIndex support.
"""
        with open(os.path.join(self.output_dir, "API_REFERENCE.md"), "w") as f:
            f.write(content)

    def _generate_deployment_guide(self):
        content = """# Deployment Guide

Differential KV is designed for sparse execution efficiency.

## VRAM Optimization
Use `--sparse-mode lowrank_sparse` for RTX 40-series cards.
Use `--sparse-mode shared_basis` for data-center A100/H100 clusters.
"""
        with open(os.path.join(self.output_dir, "DEPLOYMENT.md"), "w") as f:
            f.write(content)

if __name__ == "__main__":
    materializer = DocumentationMaterializer()
    materializer.materialize_all()
    print("Documentation materialized.")
