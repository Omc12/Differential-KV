
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import json
import time
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.iso_resolver import ISOResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_21_2_light"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_test():
    """
    PHASE 21.2: ISO ULTRA-LIGHTWEIGHT VALIDATION.
    Optimized for rapid architectural iteration.
    Target: < 5-8 minutes.
    """
    # Load Model ONCE
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    model = loader.load(attn_implementation="sdpa")
    suite = SPSPrecisionSuite(tokenizer)

    # 6 focused object integrity tests (16-32 tokens max)
    tests = [
        {"name": "ISO Registration", "domain": "api_key_complex", "len": 16},
        {"name": "Topology Hashing", "domain": "hex_sequence", "len": 16},
        {"name": "Object Persistence", "domain": "structured_id", "len": 32},
        {"name": "Lineage Continuity", "domain": "json_exact", "len": 16},
        {"name": "Mutation Suppression", "domain": "activation_code", "len": 32},
        {"name": "Serialization Integrity", "domain": "adversarial_delimiters", "len": 32},
    ]

    print(f"Starting Phase 21.2 ISO ULTRA-LIGHT Batch Smoke Test ({len(tests)} tests)...")
    print("Policy: FAST ITERATION + OBJECT INTEGRITY")
    total_start = time.time()
    
    for idx, test in enumerate(tests):
        print(f"[{idx+1}/{len(tests)}] {test['name']}...")
        
        # New resolver instance for clean ISO state
        resolver = ISOResolver(tokenizer)

        # Setup Case (4096 context ONLY)
        test_case = suite.create_case(test["domain"], 4096, target_len=test["len"])
        input_ids = torch.tensor([test_case["tokens"]], device=model.device)
        needle = test_case["needle"]

        # Reset VRAM & Generation
        torch.cuda.empty_cache()
        past_key_values = DynamicCache()
        
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
            # Register ISOs from prefill context
            resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), input_ids)
            logits = outputs.logits[:, -1, :].float()
            del outputs

        generated_tokens = []
        gen_start = time.time()
        for i in range(test["len"] + 8):
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
                resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), torch.tensor([[token_id]], device=model.device))
                logits = outputs.logits[:, -1, :].detach().float()
                del outputs
            
            if token_id == tokenizer.eos_token_id: break

        # Evaluation
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        exact_match = needle.lower() in output_text.lower()
        stats = resolver.get_iso_summary()
        
        # ISO Telemetry Summary
        summary = {
            "test": test["name"],
            "exact_match": exact_match,
            "object_integrity": stats.get("object_integrity", 1.0),
            "topology_stability": stats.get("topology_stability", 1.0),
            "lineage_depth": stats.get("lineage_depth", 0),
            "registered_objects": stats.get("registered_objects", 0),
            "recall_precision": stats.get("mean_legitimacy", 0),
        }
        
        with open(os.path.join(RESULTS_DIR, "iso_integrity_summary.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"  Result: EM={exact_match} ISOs={summary['registered_objects']} Integrity={summary['object_integrity']:.3f}")

    duration = (time.time() - total_start) / 60
    print(f"ISO ULTRA-LIGHT Test Complete. Total duration: {duration:.2f} minutes.")

if __name__ == "__main__":
    run_test()
