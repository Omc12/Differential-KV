
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.srl_resolver import SRLResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_21_1_light"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 21.1: ULTRA-LIGHTWEIGHT LEGITIMACY VALIDATION.
    Optimized for fast architectural iteration.
    Target: < 5-10 minutes.
    """
    # Load Model ONCE
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    # SDPA for maximum speed; legitimacy doesn't strictly require attention maps
    model = loader.load(attn_implementation="sdpa")
    suite = SPSPrecisionSuite(tokenizer)

    # Focused legitimacy tests (6 tests, 16-32 tokens max)
    tests = [
        {"name": "Correct Recall", "domain": "api_key_complex", "len": 16},
        {"name": "False Recall Suppression", "domain": "hex_sequence", "len": 16},
        {"name": "Multi-candidate Routing", "domain": "structured_id", "len": 32},
        {"name": "Entropy Preservation", "domain": "json_exact", "len": 16},
        {"name": "Delayed Recall", "domain": "activation_code", "len": 32},
        {"name": "Confusion Stress", "domain": "adversarial_delimiters", "len": 32},
    ]

    print(f"Starting Phase 21.1 ULTRA-LIGHT Batch Smoke Test ({len(tests)} tests)...")
    print("Policy: FAST ITERATION + MINIMAL TELEMETRY")
    total_start = time.time()
    
    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        
        # New resolver instance for clean metrics per test
        resolver = SRLResolver(tokenizer)

        # Setup Case (4096 context ONLY)
        test_case = suite.create_case(test["domain"], 4096, target_len=test["len"])
        input_ids = torch.tensor([test_case["tokens"]], device=model.device)
        needle = test_case["needle"]

        # Reset VRAM & Generation
        torch.cuda.empty_cache()
        past_key_values = DynamicCache()
        
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            # Register hubs from prefill context
            resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), input_ids)
            logits = outputs.logits[:, -1, :].float()
            del outputs

        generated_tokens = []
        gen_start = time.time()
        for i in range(test["len"] + 8): # Minimal buffer
            # Legitimacy-aware guidance
            logits = resolver.guide_decoder(logits, None)
            
            token_id = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(token_id)
            
            resolver.record_generated_token(token_id, logits.detach().cpu())
                
            with torch.no_grad():
                outputs = model(
                    torch.tensor([[token_id]], device=model.device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                # Prune and resolve state
                resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), torch.tensor([[token_id]], device=model.device))
                logits = outputs.logits[:, -1, :].detach().float()
                del outputs
            
            if token_id == tokenizer.eos_token_id: break

        # Minimal Evaluation
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        exact_match = needle.lower() in output_text.lower()
        stats = resolver.get_srl_summary()
        
        # Light Telemetry Summary
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "recall_precision": stats.get("mean_legitimacy", 0),
            "false_recall_rate": stats.get("false_recall_rate", 0),
            "entropy_health": stats.get("entropy_preservation", 0),
            "reinjection_strength": stats.get("reinjection_strength", 0),
        }
        
        with open(os.path.join(RESULTS_DIR, "srl_legitimacy_summary.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} Legit={summary['recall_precision']:.3f} FalseRecall={summary['false_recall_rate']:.3f}")

    duration = (time.time() - total_start) / 60
    print(f"ULTRA-LIGHT Test Complete. Total duration: {duration:.2f} minutes.")

if __name__ == "__main__":
    run_test()
