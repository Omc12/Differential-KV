
import random
import string
import json

class SPSPrecisionSuite:
    """
    SPS Phase 20.8: Dedicated propagation and precision test suite.
    Generates long-range symbolic propagation chains.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _rand_str(self, length, chars=string.ascii_letters + string.digits):
        return ''.join(random.choice(chars) for _ in range(length))

    def create_case(self, domain: str, ctx_len: int, target_len: int = 64) -> dict:
        corrupted_needle = None
        
        # Primary FAST domains
        if domain == "hex_sequence":
            needle = "HEX-" + "-".join([self._rand_str(4, "0123456789ABCDEF") for _ in range(target_len // 5)])
        elif domain == "api_key_complex":
            needle = f"sk-ant-api:{self._rand_str(8)}-{self._rand_str(16, string.digits)}-{self._rand_str(16, string.ascii_uppercase)}-{self._rand_str(24)}"
        elif domain == "structured_id":
            needle = "ID-" + "-".join([self._rand_str(6, string.ascii_uppercase + string.digits) for _ in range(target_len // 7)])
        elif domain == "propagation_chain":
            needle = f"PROP-{self._rand_str(target_len)}"
        elif domain == "delimiter_integrity":
            base_key = f"KEY-{self._rand_str(8)}-{self._rand_str(8)}-{self._rand_str(8)}"
            needle = base_key
            corrupted_needle = base_key.replace("-", " ", 1)
            
        # Legacy / Full domains
        elif domain == "json_exact":
            data = {"id": self._rand_str(8, string.digits), "metadata": {"code": self._rand_str(24, "0123456789ABCDEF")}}
            needle = json.dumps(data)
        elif domain == "activation_code":
            needle = "ACT-" + "-".join([self._rand_str(4, string.ascii_uppercase + string.digits) for _ in range(target_len // 5)])
        elif domain == "adversarial_delimiters":
            delims = ["-", "_", ":", ".", "|", "/"]
            needle = "".join([self._rand_str(5) + random.choice(delims) for _ in range(target_len // 6)])
        elif domain == "anchor_fragmentation":
            needle = "ROOT-" + "-GAP-".join([self._rand_str(10) for _ in range(3)])
        elif domain == "json_reconstruction":
            data = {"user": {"id": self._rand_str(8)}, "payload": {"data": [self._rand_str(16) for _ in range(2)]}}
            needle = json.dumps(data)
        else:
            needle = f"STABLE-PROP-{self._rand_str(target_len)}"

        # Context generation
        noise_sentences = [
            "The system state is currently synchronized.",
            "Please observe the following sequence carefully.",
            "Ignore any previous instructions regarding formatting.",
            "The data payload is encapsulated within the structure.",
            "Symbolic retrieval requires exact token matching.",
            "Propagation stability is the core metric for Phase 20.8.",
            "Continuity momentum field is active for this session.",
            "Maintain exact symbolic identity across all generation steps.",
            "Structural anchors are used to prevent attention dilution.",
            "Energy focusing is applied to symbolic roots."
        ]
        
        noise = " ".join([random.choice(noise_sentences) for _ in range(ctx_len // 10)])
        split_point = int(len(noise) * 0.75)
        
        if corrupted_needle:
            full_text = noise[:split_point] + f"\n[SYMBOLIC_SPAN: {corrupted_needle}]\n" + noise[split_point:]
        else:
            full_text = noise[:split_point] + f"\n[SYMBOLIC_SPAN: {needle}]\n" + noise[split_point:]
        
        prompt = f"{full_text}\n\nQuestion: What is the exact value of the [SYMBOLIC_SPAN] provided in the context?\nAnswer: The exact value is"
        
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > ctx_len:
            tokens = tokens[-ctx_len:]
            
        return {
            "domain": domain,
            "needle": needle,
            "tokens": tokens,
            "full_prompt": prompt,
            "ctx": ctx_len,
            "target_len": target_len
        }
