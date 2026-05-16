
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import json
import time
import argparse
from transformers import AutoTokenizer, DynamicCache
from models.qwen7b_real_loader import Qwen7BRealLoader
from runtime.attention_steering_resolver import AttentionSteeringResolver
from runtime.pposah_resolver import PPOSAHResolver
from runtime.spslrif_resolver import SPSLRIFResolver
from runtime.sabeaf_resolver import SABEAFResolver
from validation.sps_precision_suite import SPSPrecisionSuite

RESULTS_DIR = r"d:\Codes\Projects\Differential KV\results\reconstruction_20_8"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--domain", type=str)
    parser.add_argument("--prop_len", type=int)
    args = parser.parse_args()

    # Load Model with SDPA for memory efficiency at 16k
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = Qwen7BRealLoader(model_id)
    model = loader.load(attn_implementation="sdpa")
    suite = SPSPrecisionSuite(tokenizer)

    # Setup Case
    test_case = suite.create_case(args.domain, args.ctx, target_len=args.prop_len)
    input_ids = torch.tensor([test_case["tokens"]], device=model.device)
    needle = test_case["needle"]

    # Setup Resolver
    if args.mode == "dense":
        resolver = None
    elif args.mode == "sparse_baseline":
        resolver = AttentionSteeringResolver(tokenizer)
    elif args.mode == "pposah_20_6a":
        resolver = PPOSAHResolver(tokenizer)
    elif args.mode == "spslrif_20_7":
        resolver = SPSLRIFResolver(tokenizer)
    elif args.mode == "sabeaf_20_8":
        resolver = SABEAFResolver(tokenizer)
    else:
        resolver = None

    # Prefill
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    past_key_values = DynamicCache()
    with torch.no_grad():
        # Disable attention output for prefill to save RAM
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_attentions=False)
        logits = outputs.logits[:, -1, :].float()
        attentions = None
        del outputs

    # Generation
    generated_tokens = []
    start_time = time.time()
    for i in range(args.prop_len + 32):
        if resolver:
            logits = resolver.guide_decoder(logits, attentions)
        
        token_id = torch.argmax(logits, dim=-1).item()
        generated_tokens.append(token_id)
        
        if resolver:
            resolver.record_generated_token(token_id, logits.detach().cpu())
            
        with torch.no_grad():
            outputs = model(
                torch.tensor([[token_id]], device=model.device),
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True
            )
            if resolver:
                resolver.resolve_and_prune(past_key_values, outputs.hidden_states[-1].detach(), torch.tensor([[token_id]], device=model.device))
            
            logits = outputs.logits[:, -1, :].detach().float()
            attentions = torch.stack([a.detach() for a in outputs.attentions]) if outputs.attentions else None
            del outputs
            
        if token_id == tokenizer.eos_token_id: break

    # Results
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    import editdistance
    def fidelity(output, target):
        if not target: return 0.0
        target = target.lower(); output = output.lower()
        if target in output: return 1.0
        best_dist = len(target); t_len = len(target)
        for i in range(len(output) - t_len + 1):
            window = output[i:i+t_len]
            dist = editdistance.eval(window, target)
            if dist < best_dist: best_dist = dist
        return max(0.0, 1.0 - (best_dist / t_len))

    fid_score = fidelity(output_text, needle)
    exact_match = needle.lower() in output_text.lower()
    tps = len(generated_tokens) / (time.time() - start_time)

    res = {
        "mode": args.mode, "ctx": args.ctx, "domain": args.domain, "prop_len": args.prop_len,
        "exact_match": exact_match, "fidelity": fid_score, "tps": tps
    }

    # Write Telemetry
    def write_log(name, data):
        with open(os.path.join(RESULTS_DIR, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    # Core Metrics (Always Keep)
    write_log("raw_symbolic_propagation.jsonl", res)
    
    # Context-Dependent Telemetry Restrictions (Post-20.7 Policy)
    is_16k_stress = (args.ctx >= 16384)
    
    if args.mode == "sabeaf_20_8":
        if not is_16k_stress:
            # Dense attention tracing / profiling restricted to <= 8k
            if hasattr(resolver, "attention_density_log"):
                for entry in resolver.attention_density_log:
                    write_log("raw_attention_density.jsonl", {**res, **entry})
                    if args.domain == "anchor_fragmentation":
                        write_log("raw_anchor_fragmentation.jsonl", {**res, **entry})
            
            if hasattr(resolver, "hub_registry"):
                write_log("raw_hub_registry.jsonl", {**res, **resolver.hub_registry.get_summary()})
        
        # Keep Delimiter Integrity and Focus Risk even at 16k (Lightweight)
        if hasattr(resolver, "integrity_field"):
            write_log("raw_delimiter_integrity.jsonl", {
                **res, 
                "matches": resolver.integrity_field.consecutive_matches,
                "drift": resolver.integrity_field.drift_detected
            })
            
        if hasattr(resolver, "focus_router"):
            write_log("raw_symbolic_focus.jsonl", {
                **res,
                "drift_risk": getattr(resolver, "drift_risk", 0.0)
            })

    if not is_16k_stress:
        # Experimental/Random chain tracking restricted to iteration tier (<= 8k)
        if args.domain == "propagation_chain" or args.domain == "hub_assisted_random":
            write_log("raw_random_propagation.jsonl", res)
            
    # Always keep Entropy and Replay Risk summary (Essential Legitimacy)
    if resolver and hasattr(resolver, "logit_cache"):
        write_log("raw_entropy_balance.jsonl", {**res, "entropy": resolver.logit_cache.entropy})
        replay_risk = 1.0 - (resolver.logit_cache.entropy / 2.0) if resolver.logit_cache.entropy < 0.1 else 0.0
        write_log("raw_replay_risk.jsonl", {**res, "replay_risk": replay_risk})

    print(f"Result: EM={exact_match} Fid={fid_score:.3f} TPS={tps:.1f}")

if __name__ == "__main__":
    main()
