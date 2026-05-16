import torch
import json
from models.qwen7b_real_loader import Qwen7BRealLoader
from validation.multidomain_symbolic_suite import MultidomainSymbolicSuite
from runtime.attention_steering_resolver import AttentionSteeringResolver
from transformers import AutoTokenizer, DynamicCache

def verify_8k_minimal():
    print("[PHASE 20.4] Verifying Minimal Steering at 8k Context...")
    loader = Qwen7BRealLoader()
    model = loader.load(attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    suite = MultidomainSymbolicSuite(tokenizer)
    
    # Test case at 8k
    test_case = suite.create_domain_test_case("activation_code", 8192)
    input_ids = torch.tensor([test_case['tokens']]).to("cuda")
    needle = test_case['needle']
    
    # Use Standard Bias (15.0)
    resolver = AttentionSteeringResolver(tokenizer, anchor_budget=2048, fidelity_budget=1024)
    resolver.logit_bias_strength = 15.0
    
    past_key_values = DynamicCache()
    chunk_size = 512
    for i in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, i:i+chunk_size]
        with torch.no_grad():
            outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            legacy = past_key_values.to_legacy_cache()
            pruned, _ = resolver.resolve_and_prune(legacy, outputs.hidden_states[-1], chunk)
            past_key_values = DynamicCache.from_legacy_cache(pruned)
            
    curr_input = input_ids[:, -1:]
    generated_tokens = []
    for _ in range(32):
        with torch.no_grad():
            outputs = model(input_ids=curr_input, past_key_values=past_key_values, use_cache=True, output_attentions=True)
            logits = outputs.logits[:, -1, :]
            attentions = torch.stack(outputs.attentions)
            logits = resolver.guide_decoder(logits, attentions)
            token = torch.argmax(logits, dim=-1).unsqueeze(0)
            generated_tokens.append(token.item())
            curr_input = token
            if token.item() == tokenizer.eos_token_id: break
            
    output_text = tokenizer.decode(generated_tokens)
    success = needle.lower() in output_text.lower()
    print(f"[RESULT] Success: {success}")
    print(f"OUTPUT: {output_text}")

if __name__ == "__main__":
    verify_8k_minimal()
