import random

class RealWorldContextGenerator:
    """Generates realistic context workloads (conversations, codebases)."""
    
    def generate_conversation(self, length_tokens, needle):
        dialogue = [
            "User: Hello, I need to check the system logs.",
            "Assistant: Sure, I can help with that. Which log specifically?",
            "User: The one regarding the Differential KV initialization.",
            "Assistant: Found it. It seems the calibration was successful.",
            f"Assistant: The record shows the secure token was {needle}.",
            "User: Great. What about the memory usage?",
            "Assistant: VRAM usage is stable at 6.7GB.",
            "User: Any errors?",
            "Assistant: No errors detected in the current session."
        ]
        
        filler = "User: Can you repeat that? Assistant: Yes, the system is stable. " * 20
        # Interleave dialogue with filler to reach length
        full_text = ". ".join(dialogue)
        while len(full_text.split()) < length_tokens // 5: # rough estimation
            full_text += filler
            
        return full_text

    def generate_codebase_context(self, length_tokens, needle):
        code_blocks = [
            "def initialize_system():\n    print('Initializing...')",
            "class MemoryResolver:\n    def __init__(self):\n        pass",
            f"SECRET_KEY = '{needle}'",
            "def resolve_kv(cache, hidden_states):\n    # Sparse logic here\n    return cache"
        ]
        
        filler = "import torch\nimport os\n# Helper function to track VRAM\ndef track_vram():\n    pass\n" * 10
        full_text = "\n".join(code_blocks)
        while len(full_text.split()) < length_tokens // 5:
            full_text += filler
            
        return full_text
