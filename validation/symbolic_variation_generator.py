import random
import string
import uuid

class SymbolicVariationGenerator:
    """Generates randomized symbolic structures for cross-domain validation."""
    
    def generate_activation_code(self):
        # Example: SIGMA-19-6-SADACGG-TEST
        prefix = random.choice(["SIGMA", "ALPHA", "OMEGA", "DELTA", "THETA"])
        parts = [str(random.randint(0, 99)) for _ in range(2)]
        suffix = ''.join(random.choices(string.ascii_uppercase, k=7))
        return f"{prefix}-{parts[0]}-{parts[1]}-{suffix}-CODE"

    def generate_api_key(self):
        # Example: sk-ant-api03-abcdef123456...
        prefix = "sk-ant-api"
        version = f"{random.randint(0, 9):02d}"
        body = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
        return f"{prefix}{version}-{body}"

    def generate_structured_id(self):
        # Example: UUID or similar
        return str(uuid.uuid4())

    def generate_json_snippet(self):
        # Example: {"id": "...", "status": "active"}
        uid = str(uuid.uuid4())[:8]
        status = random.choice(["active", "pending", "suspended", "verified"])
        return f'{{"id": "{uid}", "status": "{status}", "type": "symbolic_anchor"}}'

    def generate_code_fragment(self):
        # Example: const_val = 0xABCDEF
        name = ''.join(random.choices(string.ascii_lowercase, k=8))
        val = hex(random.randint(0x100000, 0xFFFFFF)).upper()
        return f"const {name}_SECRET = {val};"

    def apply_variation(self, text):
        """Applies formatting, casing, and spacing variations."""
        strategy = random.choice(["none", "lowercase", "uppercase", "extra_spaces", "brackets"])
        
        if strategy == "lowercase":
            return text.lower()
        elif strategy == "uppercase":
            return text.upper()
        elif strategy == "extra_spaces":
            # Add random spaces around separators
            if "-" in text:
                return text.replace("-", " - ")
            return f"  {text}  "
        elif strategy == "brackets":
            return f"[{text}]"
        return text

    def get_random_domain_needle(self):
        generators = [
            self.generate_activation_code,
            self.generate_api_key,
            self.generate_structured_id,
            self.generate_json_snippet,
            self.generate_code_fragment
        ]
        needle = random.choice(generators)()
        return self.apply_variation(needle)
