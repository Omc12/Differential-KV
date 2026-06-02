import os
import sys
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.hf_diffkv_wrapper import DiffKVHFWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    LARGE_PROMPT_PAPER = """
Abstract
While there is a growing effort towards AI for Sustainability (e.g. towards the sustainable development goals) it is time to move beyond that and to address the sustainability of developing and using AI systems. In this paper I propose a definition of Sustainable AI; Sustainable AI is a movement to foster change in the entire lifecycle of AI products (i.e. idea generation, training, re-tuning, implementation, governance, and post-use disposal) towards ecological and social sustainability. Sustainable AI is divided into two categories: AI for sustainability (using AI to support sustainability goals) and sustainability of AI (sustainable development, training, and use of AI). The focus of this paper is on the latter.
In particular, I argue that the current trajectory of AI development and use (characterized by massive deep learning models requiring huge amounts of energy and resources to train and run) is unsustainable. I analyze the ecological and social impacts of the AI lifecycle, including resource extraction for hardware, greenhouse gas emissions from data centers during training and inference, and the social inequalities perpetuated by high compute costs. Finally, I propose a set of guiding principles and actionable recommendations for researchers, developers, and policymakers to transition towards a sustainable AI ecosystem. These include energy-efficient hardware, green software engineering, open data and models, and robust governance frameworks that incorporate environmental impact assessments.
"""
    long_abstract = "\n".join([f"Section {i+1}:\n{LARGE_PROMPT_PAPER}" for i in range(10)])
    prompt = f"<|im_start|>user\nHere is a long research text:\n{long_abstract}\n\nBased on the text above, summarize the key points of Sustainable AI in 3 bullet points.<|im_end|>\n<|im_start|>assistant\n"
    
    print("1. Loading baseline model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model_baseline = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
    )
    model_baseline.eval()
    
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = encoded.input_ids[0].tolist()
    prefill_len = len(prompt_ids)
    
    print(f"Prefill length: {prefill_len} tokens")
    
    # Run baseline prefill step
    print("Running baseline prefill...")
    with torch.no_grad():
        outputs_baseline = model_baseline(input_ids=encoded.input_ids, use_cache=True)
    past_kv_baseline = outputs_baseline.past_key_values
    logits_b = outputs_baseline.logits[0, -1, :].cpu().float()
    
    # -----------------------------------------------------------------
    # Load patched model
    # -----------------------------------------------------------------
    print("\n2. Loading patched model with current active runtime configuration...")
    wrapper = DiffKVHFWrapper(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        config={"rank": 32, "micro_block_size": 32},
        device=device,
    )
    
    print("Running patched model prefill...")
    session_id = "default"
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.model._diffkv_session_ids = [session_id]
    
    with torch.no_grad():
        outputs_patched = wrapper.model(input_ids=encoded.input_ids, use_cache=True)
        wrapper.manager.compress_prefill_kv(session_id)
    logits_p = outputs_patched.logits[0, -1, :].cpu().float()
    
    # Compare first logits
    diff = (logits_p - logits_b).abs()
    print(f"\n--- Prefill/Step 0 comparison: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f} ---")
    
    # We will do teacher forced generation comparison for 20 steps
    # We feed the baseline's chosen tokens to BOTH models.
    print("\n============================================================")
    print("  TEST 1: Teacher Forcing (feeding identical tokens)")
    print("============================================================")
    
    cur_pos = prefill_len
    past_b = past_kv_baseline
    past_p = None # DiffKV handles internally
    
    teacher_tokens = []
    
    # Generate 20 tokens step by step
    for step in range(20):
        # Determine next token using baseline logits
        next_id_b = logits_b.argmax().item()
        teacher_tokens.append(next_id_b)
        
        # Determine next token using patched logits
        next_id_p = logits_p.argmax().item()
        
        token_str_b = tokenizer.decode([next_id_b])
        token_str_p = tokenizer.decode([next_id_p])
        
        diff = (logits_p - logits_b).abs()
        
        print(f"Step {step+1:02d} (pos={cur_pos}):")
        print(f"  Baseline token: {next_id_b:<6} ({token_str_b!r})")
        print(f"  Patched token : {next_id_p:<6} ({token_str_p!r})")
        print(f"  Max Diff      : {diff.max().item():.6f}")
        print(f"  Mean Diff     : {diff.mean().item():.6f}")
        
        # Run next step
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        input_ids = torch.tensor([[next_id_b]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs_b = model_baseline(input_ids=input_ids, position_ids=pos_tensor, past_key_values=past_b, use_cache=True)
            past_b = outputs_b.past_key_values
            logits_b = outputs_b.logits[0, -1, :].cpu().float()
            
            outputs_p = wrapper.model(input_ids=input_ids, position_ids=pos_tensor, past_key_values=past_p, use_cache=True)
            logits_p = outputs_p.logits[0, -1, :].cpu().float()
            
        cur_pos += 1

    # Cleanup and re-run for Autoregressive Generation
    print("\n============================================================")
    print("  TEST 2: Autoregressive Generation (independent generation)")
    print("============================================================")
    
    # Reset patched model session
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=prefill_len)
    wrapper.model._diffkv_session_ids = [session_id]
    
    # Run prefill again
    with torch.no_grad():
        outputs_patched = wrapper.model(input_ids=encoded.input_ids, use_cache=True)
        wrapper.manager.compress_prefill_kv(session_id)
    logits_p = outputs_patched.logits[0, -1, :].cpu().float()
    
    # Baseline prefill is already done, reset past_kv
    # But wait, past_kv_baseline was mutated in TEST 1, so we need to copy it or re-run baseline prefill.
    # To be safe, let's re-run baseline prefill
    # Wait, we deleted model_baseline? No, we didn't delete it, we just didn't use it.
    # Wait! In the script above: `del model_baseline` was NOT present in our new script! We kept it.
    # So we can just re-run baseline prefill.
    print("Re-running baseline prefill...")
    with torch.no_grad():
        outputs_baseline = model_baseline(input_ids=encoded.input_ids, use_cache=True)
    past_b = outputs_baseline.past_key_values
    logits_b = outputs_baseline.logits[0, -1, :].cpu().float()
    
    cur_pos = prefill_len
    past_p = None
    
    generated_b = []
    generated_p = []
    
    for step in range(25):
        next_id_b = logits_b.argmax().item()
        next_id_p = logits_p.argmax().item()
        
        generated_b.append(next_id_b)
        generated_p.append(next_id_p)
        
        token_str_b = tokenizer.decode([next_id_b])
        token_str_p = tokenizer.decode([next_id_p])
        
        print(f"Step {step+1:02d}:")
        print(f"  Baseline: {next_id_b:<6} ({token_str_b!r}) -> Full text so far: {tokenizer.decode(generated_b)!r}")
        print(f"  Patched : {next_id_p:<6} ({token_str_p!r}) -> Full text so far: {tokenizer.decode(generated_p)!r}")
        
        # Update baseline
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        input_ids_b = torch.tensor([[next_id_b]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs_b = model_baseline(input_ids=input_ids_b, position_ids=pos_tensor, past_key_values=past_b, use_cache=True)
            past_b = outputs_b.past_key_values
            logits_b = outputs_b.logits[0, -1, :].cpu().float()
            
        # Update patched
        input_ids_p = torch.tensor([[next_id_p]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs_p = wrapper.model(input_ids=input_ids_p, position_ids=pos_tensor, past_key_values=past_p, use_cache=True)
            logits_p = outputs_p.logits[0, -1, :].cpu().float()
            
        cur_pos += 1

    wrapper.stop()

if __name__ == "__main__":
    main()
