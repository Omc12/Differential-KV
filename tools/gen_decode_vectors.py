#!/usr/bin/env python3
import struct
import numpy as np
import os

def write_bin(filename, data_dict):
    with open(filename, "wb") as f:
        f.write(struct.pack("<I", len(data_dict)))
        for name, arr in data_dict.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            
            dtype_bytes = str(arr.dtype).encode("utf-8")
            f.write(struct.pack("<I", len(dtype_bytes)))
            f.write(dtype_bytes)
            
            f.write(struct.pack("<I", arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            
            data_bytes = arr.tobytes()
            f.write(struct.pack("<Q", len(data_bytes)))
            f.write(data_bytes)

def main():
    # Set random seed for determinism
    np.random.seed(42)
    
    # Parameters
    H_q = 8
    H_kv = 2
    D = 64
    rank = 32;
    S_max = 64
    n_slots = 4
    K_active = 3
    scale = 0.125
    has_rope = False
    rope_freq_base = 1000000.0
    approximate_attn = True
    MAX_RESIDUAL = 128
    
    # Inputs
    Q = np.random.randn(H_q, D).astype(np.float32) * 0.1
    slots = np.array([0, 1, 2], dtype=np.int32)
    
    # Pool arrays
    host_U = np.random.randint(-100, 100, size=(n_slots, S_max, rank), dtype=np.int8)
    host_U_scale = (np.random.rand(n_slots) * 0.1).astype(np.float16)
    host_U_row_scale = (np.random.rand(n_slots, S_max) * 0.1).astype(np.float16)
    
    host_VK = (np.random.randn(n_slots, rank, H_kv, D) * 0.1).astype(np.float16)
    host_VV = (np.random.randn(n_slots, rank, H_kv, D) * 0.1).astype(np.float16)
    
    host_anchors_K = (np.random.randn(n_slots, H_kv, D) * 0.1).astype(np.float16)
    host_anchors_V = (np.random.randn(n_slots, H_kv, D) * 0.1).astype(np.float16)
    
    host_scales = (np.random.rand(n_slots) * 0.1 + 0.9).astype(np.float16)
    host_valid_mask = np.zeros((n_slots, S_max), dtype=np.float16)  # 0 means valid
    
    host_seq_lens = np.array([S_max, S_max, S_max // 2, 0], dtype=np.int32)
    host_anchor_positions = np.array([0, 256, 512, 0], dtype=np.int32)
    
    # Residuals
    host_res_K_pos = np.full((n_slots, MAX_RESIDUAL), -1, dtype=np.int32)
    host_res_V_pos = np.full((n_slots, MAX_RESIDUAL), -1, dtype=np.int32)
    
    # Plant a few residuals
    for s in range(n_slots):
        slen = host_seq_lens[s]
        if slen > 0:
            for ri in range(5):
                t = ri * 4
                if t < slen:
                    host_res_K_pos[s, ri] = t
                    host_res_V_pos[s, ri] = t
                    
    host_res_K_val = (np.random.randn(n_slots, MAX_RESIDUAL, H_kv, D) * 0.05).astype(np.float16)
    host_res_V_val = (np.random.randn(n_slots, MAX_RESIDUAL, H_kv, D) * 0.05).astype(np.float16)
    
    # Run the attention reference math in Python
    # We will compute expected_out [H_q, D] and expected_lse [H_q]
    expected_out = np.zeros((H_q, D), dtype=np.float32)
    expected_lse = np.zeros(H_q, dtype=np.float32)
    
    g = H_q // H_kv
    
    # We only process slots that are actually active/routed and CompressedResident
    # Since all slots 0, 1, 2 are in `slots` and we set state to CompressedResident in C++,
    # we run the math for slot_id in [0, 1, 2]
    active_slots = [0, 1, 2]
    
    for h in range(H_q):
        kv_head = h // g
        
        # Calculate scores for active slots
        max_score = -1e30
        sum_exp = 0.0
        
        # Store intermediate scores per slot
        slot_token_scores = {}
        slot_anchor_score = {}
        
        for slot_id in active_slots:
            slen = host_seq_lens[slot_id]
            
            # Anchor score
            anc_K_float = host_anchors_K[slot_id, kv_head].astype(np.float32)
            score_anc = np.dot(Q[h], anc_K_float)
            slot_anchor_score[slot_id] = score_anc
            
            if score_anc * scale > max_score:
                max_score = score_anc * scale
                
            # Delta scores
            q_proj = np.zeros(rank, dtype=np.float32)
            for r in range(rank):
                vkr_float = host_VK[slot_id, r, kv_head].astype(np.float32)
                q_proj[r] = np.dot(Q[h], vkr_float)
                
            token_scores = []
            for t in range(slen):
                delta = 0.0
                u_row = host_U[slot_id, t]
                for r in range(rank):
                    delta += q_proj[r] * float(u_row[r])
                    
                # Residual score
                res_score = 0.0
                # Find if t has a residual
                ri = -1
                for idx in range(MAX_RESIDUAL):
                    if host_res_K_pos[slot_id, idx] == t:
                        ri = idx
                        break
                if ri != -1:
                    rk_val = host_res_K_val[slot_id, ri, kv_head].astype(np.float32)
                    res_score = np.dot(Q[h], rk_val)
                    
                ku = float(host_U_row_scale[slot_id, t].astype(np.float32))
                blk_sc = float(host_scales[slot_id].astype(np.float32))
                t_score = (delta * ku * blk_sc + res_score + score_anc) * scale
                token_scores.append(t_score)
                if t_score > max_score:
                    max_score = t_score
                    
            slot_token_scores[slot_id] = token_scores
            
        # Sum exp
        for slot_id in active_slots:
            slen = host_seq_lens[slot_id]
            sum_exp += np.exp(slot_anchor_score[slot_id] * scale - max_score)
            for t in range(slen):
                sum_exp += np.exp(slot_token_scores[slot_id][t] - max_score)
                
        expected_lse[h] = max_score + np.log(max(sum_exp, 1e-9))
        
        # Accumulate output values
        accum = np.zeros(D, dtype=np.float64)
        for slot_id in active_slots:
            slen = host_seq_lens[slot_id]
            blk_sc = float(host_scales[slot_id].astype(np.float32))
            
            w_anc = np.exp(slot_anchor_score[slot_id] * scale - max_score) / sum_exp
            sum_w = 0.0
            w_proj = np.zeros(rank, dtype=np.float64)
            res_v_accum = np.zeros(D, dtype=np.float64)
            
            for t in range(slen):
                w_t = np.exp(slot_token_scores[slot_id][t] - max_score) / sum_exp
                sum_w += w_t
                
                u_row = host_U[slot_id, t]
                ku = float(host_U_row_scale[slot_id, t].astype(np.float32))
                for r in range(rank):
                    w_proj[r] += w_t * float(u_row[r]) * ku
                    
                # Residual V
                ri = -1
                for idx in range(MAX_RESIDUAL):
                    if host_res_V_pos[slot_id, idx] == t:
                        ri = idx
                        break
                if ri != -1:
                    rv = host_res_V_val[slot_id, ri, kv_head].astype(np.float32)
                    res_v_accum += w_t * rv
                    
            w_total = w_anc + sum_w
            svd_v = np.zeros(D, dtype=np.float64)
            for r in range(rank):
                wr = w_proj[r] * blk_sc
                vvr = host_VV[slot_id, r, kv_head].astype(np.float32)
                svd_v += wr * vvr
                
            av = host_anchors_V[slot_id, kv_head].astype(np.float32)
            val = w_total * av + svd_v + res_v_accum
            accum += val
            
        expected_out[h] = accum.astype(np.float32)
        
    # Serialize data dict
    data = {
        "Q": Q,
        "slots": slots,
        "host_U": host_U,
        "host_U_scale": host_U_scale.view(np.uint16),
        "host_U_row_scale": host_U_row_scale.view(np.uint16),
        "host_VK": host_VK.view(np.uint16),
        "host_VV": host_VV.view(np.uint16),
        "host_anchors_K": host_anchors_K.view(np.uint16),
        "host_anchors_V": host_anchors_V.view(np.uint16),
        "host_scales": host_scales.view(np.uint16),
        "host_valid_mask": host_valid_mask.view(np.uint16),
        "host_seq_lens": host_seq_lens,
        "host_anchor_positions": host_anchor_positions,
        "host_res_K_pos": host_res_K_pos,
        "host_res_V_pos": host_res_V_pos,
        "host_res_K_val": host_res_K_val.view(np.uint16),
        "host_res_V_val": host_res_V_val.view(np.uint16),
        "expected_out": expected_out,
        "expected_lse": expected_lse
    }
    
    os.makedirs("tools", exist_ok=True)
    write_bin("tools/conformance_vectors.bin", data)
    print("Golden vectors generated successfully at tools/conformance_vectors.bin")

if __name__ == "__main__":
    main()
