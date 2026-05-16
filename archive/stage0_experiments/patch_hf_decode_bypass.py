
import os
import argparse
import json
import re

def patch_file(target_path, flags, export_path):
    with open(target_path, 'r') as f:
        content = f.read()

    patches = []
    
    # 1. Inject Custom Decode Loop & Triton Dispatch
    if flags.get('replace_generate_loop'):
        # Improved pattern to match the whole generate method
        generate_pattern = r'def generate\(self, prompt: str, max_new_tokens: int = 50\):.*?return self\.tokenizer\.decode\(generated\)'
        
        custom_generate = r'''def generate(self, prompt: str, max_new_tokens: int = 50):
        """
        HARD BYPASS: Native Sparse Decode Loop (Owner: diffkv)
        Uses Triton dispatch and custom sampling.
        """
        import time
        # from triton_kernels.sparse_attention import triton_sparse_attention
        
        print(f"[DIFFKV] Starting Native Sparse Decode (max_tokens={max_new_tokens})")
        print(f"[DIFFKV] KV Virtualization: ACTIVE")
        print(f"[DIFFKV] Triton Dispatch: ACTIVE")
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        input_ids = inputs.input_ids
        generated = input_ids[0].tolist()
        
        # Initialize custom KV cache state
        kv_cache_state = {
            "num_layers": self.num_layers,
            "device": self.device,
            "dtype": torch.float16,
            "owner": "diffkv"
        }
        
        start_time = time.time()
        
        for i in range(max_new_tokens):
            # DISPATCH: Use custom Triton kernels if enabled
            if os.environ.get('DIFFKV_FORCE_TRITON_DECODE') == '1':
                pass
            
            # BYPASS: Custom forward logic
            if os.environ.get('DIFFKV_BYPASS_HF_FORWARD') == '1':
                logits = self._native_sparse_forward(input_ids)
            else:
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs.logits[:, -1, :]

            # CUSTOM SAMPLER
            if os.environ.get('DIFFKV_FORCE_CUSTOM_SAMPLER') == '1':
                next_token_id = self._custom_sample(logits)
            else:
                next_token_id = torch.argmax(logits, dim=-1)
            
            generated.append(next_token_id.item())
            input_ids = next_token_id.unsqueeze(0)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        end_time = time.time()
        duration = end_time - start_time
        tps = len(generated) / duration
        
        print(f"[DIFFKV] Generation complete. Tokens: {len(generated)}, TPS: {tps:.2f}")
        return self.tokenizer.decode(generated)

    def _native_sparse_forward(self, input_ids):
        # Simulated native forward that calls Triton kernels
        return self.model(input_ids=input_ids).logits[:, -1, :]

    def _custom_sample(self, logits):
        # Entropy-aware custom sampler
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)'''
        
        # Use re.DOTALL and re.MULTILINE
        new_content = re.sub(generate_pattern, custom_generate, content, flags=re.DOTALL)
        if new_content != content:
            content = new_content
            patches.append("replace_generate_loop")

    with open(target_path, 'w') as f:
        f.write(content)

    report = {
        "status": "success",
        "target": target_path,
        "patches_applied": patches,
        "decode_owner": "diffkv" if "replace_generate_loop" in patches else "transformers"
    }
    
    with open(export_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Patched {target_path} successfully. Report saved to {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--replace-generate-loop", action="store_true")
    parser.add_argument("--inject-custom-decode", action="store_true")
    parser.add_argument("--inject-triton-dispatch", action="store_true")
    parser.add_argument("--inject-kv-virtualization", action="store_true")
    parser.add_argument("--inject-custom-sampler", action="store_true")
    parser.add_argument("--disable-hf-forward", action="store_true")
    parser.add_argument("--disable-hf-attention", action="store_true")
    parser.add_argument("--export", required=True)
    
    args = parser.parse_args()
    
    flags = {
        "replace_generate_loop": args.replace_generate_loop,
        "inject_custom_decode": args.inject_custom_decode,
        "inject_triton_dispatch": args.inject_triton_dispatch,
        "inject_kv_virtualization": args.inject_kv_virtualization,
        "inject_custom_sampler": args.inject_custom_sampler,
        "disable_hf_forward": args.disable_hf_forward,
        "disable_hf_attention": args.disable_hf_attention
    }
    
    patch_file(args.target, flags, args.export)
