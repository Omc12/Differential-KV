"""
release/model_patch_recipes.py

Curated patch recipes for major transformer architectures.
Optimized for performance, memory, and reasoning retention.
"""

PATCH_RECIPES = {
    "Qwen/Qwen2-7B-Instruct": {
        "sparse_ratio": 0.1,
        "geometric_stabilization": True,
        "resonance_coupling": 0.8,
        "anchor_period": 512,
        "mode": "differential"
    },
    "meta-llama/Meta-Llama-3-8B": {
        "sparse_ratio": 0.15,
        "geometric_stabilization": True,
        "resonance_coupling": 0.7,
        "anchor_period": 1024,
        "mode": "differential"
    },
    "mistralai/Mistral-7B-v0.1": {
        "sparse_ratio": 0.08,
        "geometric_stabilization": True,
        "resonance_coupling": 0.9,
        "anchor_period": 256,
        "mode": "differential"
    }
}

def get_recipe(model_id: str):
    return PATCH_RECIPES.get(model_id, PATCH_RECIPES["Qwen/Qwen2-7B-Instruct"])

if __name__ == "__main__":
    import json
    print(json.dumps(PATCH_RECIPES, indent=4))
