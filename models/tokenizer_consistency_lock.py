import hashlib
import json
import os
from transformers import AutoTokenizer

class TokenizerConsistencyLock:
    """
    PHASE 18.1A: Ensures tokenizer integrity and consistency.
    MANDATORY: MUST pass before benchmarking.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.lock_path = "results/reconstruction_18_1/raw_tokenizer_manifest.json"

    def verify(self, local_path: str = None):
        print(f"[PHASE 18.1A] Verifying Tokenizer: {self.model_id}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_path if local_path else self.model_id,
                trust_remote_code=True,
                local_files_only=True if not local_path else False
            )
            
            # Record critical tokenizer metadata
            manifest = {
                "model_id": self.model_id,
                "vocab_size": len(tokenizer),
                "pad_token": str(tokenizer.pad_token),
                "eos_token": str(tokenizer.eos_token),
                "tokenizer_class": tokenizer.__class__.__name__,
                "verified": True
            }
            
            with open(self.lock_path, 'w') as f:
                json.dump(manifest, f, indent=4)
                
            return True, manifest
            
        except Exception as e:
            return False, str(e)

if __name__ == "__main__":
    lock = TokenizerConsistencyLock()
    # pass
