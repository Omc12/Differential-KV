from transformers import AutoTokenizer
import torch
import hashlib

class RealTokenizerPipeline:
    """
    Standardized tokenizer pipeline for Phase 18 benchmarks.
    Ensures identical tokenization across DiffKV and public baselines.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        print(f"[PHASE 18A] Initializing REAL tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        
        # Force padding token if missing (common in Qwen/Llama)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text: str, add_special_tokens=True):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt")

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def get_hash(self):
        """Returns a hash of the tokenizer config for reproducibility auditing."""
        config_str = str(self.tokenizer.get_vocab())
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_vocab_size(self):
        return len(self.tokenizer)

if __name__ == "__main__":
    pipeline = RealTokenizerPipeline()
    print(f"Tokenizer Hash: {pipeline.get_hash()}")
    print(f"Vocab Size: {pipeline.get_vocab_size()}")
