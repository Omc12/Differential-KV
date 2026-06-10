import os

filepath = "/Users/omchimurkar1/Desktop/Differential-KV/ACTIVE_RUNTIME/runtime/diffkv_attention.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

target = """                outputs = (attn_output,)
                if output_attentions:
                    outputs += (attn_weights,)
                if use_cache:
                    outputs += (None,)
                return outputs"""

idx1 = content.find(target)
if idx1 == -1:
    raise ValueError("Target not found")
idx2 = content.find(target, idx1 + len(target))
if idx2 == -1:
    idx_target = idx1
else:
    idx_target = idx2

replacement = """                outputs = (attn_output,)
                if output_attentions:
                    outputs += (attn_weights,)
                if use_cache:
                    outputs += (None,)

                # Reclaim VRAM on MPS during prefill
                if not is_decode and hidden_states.device.type == "mps":
                    if 'query_states' in locals(): del query_states
                    if 'key_states' in locals(): del key_states
                    if 'value_states' in locals(): del value_states
                    if 'unrot_key_states' in locals(): del unrot_key_states
                    if 'unrot_query_states' in locals(): del unrot_query_states
                    if 'attn_outputs' in locals(): del attn_outputs
                    if 'attn_output' in locals(): del attn_output
                    torch.mps.empty_cache()

                return outputs"""

content_replaced = content[:idx_target] + replacement + content[idx_target + len(target):]

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content_replaced)

print("SUCCESS")
