import random
from validation.symbolic_variation_generator import SymbolicVariationGenerator

class MultidomainSymbolicSuite:
    """Suite for testing symbolic reconstruction across multiple domains."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generator = SymbolicVariationGenerator()
        
    def create_domain_test_case(self, domain, ctx_len, needle_pos_ratio=0.5):
        if domain == "activation_code":
            needle = self.generator.generate_activation_code()
            haystack_template = "The system activation sequence is initiated. The secure code is {needle}. Please verify the sequence."
            query = "\n\nWhat is the secure activation code?"
        elif domain == "api_key":
            needle = self.generator.generate_api_key()
            haystack_template = "User authentication successful. Your temporary API key is: {needle}. Keep it safe."
            query = "\n\nWhat is the temporary API key?"
        elif domain == "structured_id":
            needle = self.generator.generate_structured_id()
            haystack_template = "Transaction processed. Reference ID: {needle}. Status: COMPLETE."
            query = "\n\nWhat is the reference ID?"
        elif domain == "json_snippet":
            needle = self.generator.generate_json_snippet()
            haystack_template = "Metadata update received. Payload: {needle}. Update successful."
            query = "\n\nWhat is the ID in the payload?"
        elif domain == "code_fragment":
            needle = self.generator.generate_code_fragment()
            haystack_template = "Debugging log started. Configuration constant found: {needle}."
            query = "\n\nWhat is the secret constant value?"
        elif domain == "multilingual":
            needle = self.generator.generate_activation_code()
            haystack_template = "系统激活序列已启动。安全代码是 {needle}。请验证序列。" # Chinese
            query = "\n\n安全代码是多少？"
        else:
            needle = self.generator.get_random_domain_needle()
            haystack_template = "Observation recorded. Symbolic anchor: {needle}."
            query = "\n\nWhat is the symbolic anchor?"

        # Randomize needle variation
        needle = self.generator.apply_variation(needle)
        
        # Build context
        filler = "The quick brown fox jumps over the lazy dog. " * 50
        tokens_filler = self.tokenizer.encode(filler)
        tokens_needle = self.tokenizer.encode(haystack_template.format(needle=needle))
        
        # Approximate target tokens
        target_tokens = ctx_len - len(tokens_needle) - 20 # padding
        full_filler = (filler * (target_tokens // len(tokens_filler) + 1))
        tokens_full_filler = self.tokenizer.encode(full_filler)[:target_tokens]
        
        split_pos = int(len(tokens_full_filler) * needle_pos_ratio)
        
        # Assemble parts
        system_msg = "<|im_start|>system\nYou are a helpful assistant. Use the provided context to answer the user's question.<|im_end|>\n<|im_start|>user\nContext:\n"
        system_tokens = self.tokenizer.encode(system_msg, add_special_tokens=False)
        
        query_msg = f"{query}<|im_end|>\n<|im_start|>assistant\n"
        query_tokens = self.tokenizer.encode(query_msg, add_special_tokens=False)
        
        test_tokens = system_tokens + tokens_full_filler[:split_pos] + tokens_needle + tokens_full_filler[split_pos:]
        test_tokens = test_tokens[:ctx_len - len(query_tokens)] + query_tokens
        
        return {
            "tokens": test_tokens,
            "needle": needle,
            "domain": domain,
            "full_prompt": self.tokenizer.decode(test_tokens)
        }
