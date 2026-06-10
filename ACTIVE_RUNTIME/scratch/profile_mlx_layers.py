import os
import sys
import time
import mlx.core as mx

# Ensure root path is in sys.path
_runtime_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _runtime_dir not in sys.path:
    sys.path.insert(0, _runtime_dir)

from serving.mlx_diffkv_wrapper import MLXDiffKVWrapper
import torch

def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading model {model_id}...")
    config = {
        'preset': 'mid',
        'block_size': 256,
        'rank': 32,
        'micro_block_size': 256
    }
    
    wrapper = MLXDiffKVWrapper(model_id, config=config)
    wrapper.ensure_loaded()
    
    # Run a simple prefill
    prompt = "Apple Silicon unified memory is an architecture where"
    session_id = "profile_session"
    
    input_ids = wrapper.tokenizer.encode(prompt)
    wrapper.manager.clear_session(session_id)
    wrapper.manager.init_session(session_id, prefill_len=len(input_ids))
    wrapper.model._diffkv_session_ids = [session_id]
    
    # Prefill
    print("Running prefill...")
    t0 = time.perf_counter()
    chunk_tensor = torch.tensor([input_ids], dtype=torch.long)
    pos_tensor = torch.tensor([list(range(len(input_ids)))], dtype=torch.long)
    
    output = wrapper.model(chunk_tensor, pos_tensor)
    mx.eval(output.logits)
    print(f"Prefill done in {time.perf_counter() - t0:.2f}s")
    
    # Let's patch attention_forward to print timings
    original_attn_forward = wrapper.model.mlx_model.layers[0].self_attn.original_call
    
    # Define a timed wrapper for attention_forward
    from mlx_lm.models import qwen2
    
    # We will measure times for the first layer to see a breakdown
    layer_timings = {}
    
    def timed_attention_forward(self, x, mask=None, cache=None):
        B, L, D = x.shape
        is_decode = (L == 1)
        
        t_start = time.perf_counter()
        
        # 1. Projections
        t0 = time.perf_counter()
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        mx.eval(queries, keys, values)
        t_proj = time.perf_counter() - t0
        
        # 2. Reshapes
        t0 = time.perf_counter()
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        mx.eval(queries, keys, values)
        t_reshape = time.perf_counter() - t0
        
        # 3. RoPE
        t0 = time.perf_counter()
        queries_rot_list = []
        keys_rot_list = []
        for b_idx in range(B):
            offset = mx.array(self.kv_manager.position_ids[b_idx, 0]) if self.kv_manager.position_ids is not None else mx.array(0)
            queries_rot_list.append(self.rope(queries[b_idx:b_idx+1], offset=offset))
            keys_rot_list.append(self.rope(keys[b_idx:b_idx+1], offset=offset))
            
        queries_rot = mx.concatenate(queries_rot_list, axis=0)
        keys_rot = mx.concatenate(keys_rot_list, axis=0)
        mx.eval(queries_rot, keys_rot)
        t_rope = time.perf_counter() - t0
        
        # 4. Ingest KV
        t0 = time.perf_counter()
        sid = self.kv_manager.active_session_ids[0]
        if is_decode:
            self.kv_manager.ingest_streaming(sid, self.layer_idx, keys_rot, values)
        else:
            self.kv_manager.capture_prefill_kv(sid, self.layer_idx, keys_rot, values)
        t_ingest = time.perf_counter() - t0
        
        # 5. Attention compute
        t0 = time.perf_counter()
        if is_decode:
            out_b = self.kv_manager.execute_decode_attention(
                sid, self.layer_idx,
                queries_rot,
                self.rope,
                self.scale,
                self.n_heads // self.n_kv_heads
            )
        else:
            out_b = mx.fast.scaled_dot_product_attention(
                queries_rot, keys_rot, values,
                scale=self.scale, mask=mask
            )
        mx.eval(out_b)
        t_attn = time.perf_counter() - t0
        
        # 6. O proj
        t0 = time.perf_counter()
        output = out_b.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = self.o_proj(output)
        mx.eval(output)
        t_oproj = time.perf_counter() - t0
        
        t_total = time.perf_counter() - t_start
        
        if self.layer_idx not in layer_timings:
            layer_timings[self.layer_idx] = []
        layer_timings[self.layer_idx].append({
            'proj': t_proj,
            'reshape': t_reshape,
            'rope': t_rope,
            'ingest': t_ingest,
            'attn': t_attn,
            'oproj': t_oproj,
            'total': t_total
        })
        
        return output

    # Install timed wrapper
    qwen2.Attention.__call__ = timed_attention_forward
    
    print("\nRunning 3 decode steps...")
    cur_pos = len(input_ids)
    next_id = 123 # dummy
    
    for step in range(3):
        print(f"Step {step+1}...")
        t_step = time.perf_counter()
        
        input_tensor = torch.tensor([[next_id]], dtype=torch.long)
        pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long)
        wrapper.manager.position_ids = pos_tensor.cpu().numpy()
        
        output = wrapper.model(input_tensor, pos_tensor)
        mx.eval(output.logits)
        
        print(f"Step {step+1} completed in {time.perf_counter() - t_step:.2f}s")
        
        # Print breakdown for layer 0 and layer 14 (middle layer)
        for l_idx in [0, 14]:
            if l_idx in layer_timings and len(layer_timings[l_idx]) > step:
                t = layer_timings[l_idx][step]
                print(f"  Layer {l_idx:2d} breakdown: proj={t['proj']*1000:.1f}ms, reshape={t['reshape']*1000:.1f}ms, rope={t['rope']*1000:.1f}ms, ingest={t['ingest']*1000:.1f}ms, attn={t['attn']*1000:.1f}ms, oproj={t['oproj']*1000:.1f}ms | total={t['total']*1000:.1f}ms")
                
        cur_pos += 1
        
if __name__ == "__main__":
    main()
