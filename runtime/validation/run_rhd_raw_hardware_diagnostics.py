import os
import sys
import time
import json
import subprocess
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
from runtime.raw_hardware_trace_system import RawHardwareTraceSystem
from runtime.scaling_integrity_guard import ScalingIntegrityGuard

def get_live_gpu_status():
    """Queries current raw GPU telemetry using nvidia-smi."""
    gpu_util = "N/A"
    power_draw = "N/A"
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if res.returncode == 0:
            parts = res.stdout.strip().split(",")
            if len(parts) >= 2:
                gpu_util = f"{parts[0].strip()}%"
                power_draw = f"{parts[1].strip()}W"
    except Exception:
        pass
    
    allocated_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return gpu_util, f"{allocated_vram_mb:.1f} MB", power_draw

def run_real_inference(
    model_id: str,
    context_target_tokens: int,
    max_new_tokens: int,
    trace_system: RawHardwareTraceSystem
):
    """
    Executes actual model inference, tracking raw hardware events, VRAM,
    and transformer layer invocations without interpretable telemetry.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} onto {device}...")
    
    # Load wrapper
    wrapper = DiffKVHFWrapper(model_id, {
        "mode": "lowrank_sparse",
        "block_size": 64,
        "rank": 16,
        "prefill_chunk_size": 512
    }, device=device)
    
    # Create prompt to reach context target
    base_text = "The quick brown fox jumps over the lazy dog. Real transformer context growth phase. "
    repetitions = (context_target_tokens // len(wrapper.tokenizer.tokenize(base_text))) + 1
    prompt = base_text * repetitions
    
    inputs = wrapper.tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs.input_ids[:, :context_target_tokens] # trim to target
    actual_seq_len = input_ids.shape[1]
    
    print(f"Executing REAL model inference. Prompt sequence length: {actual_seq_len} tokens")
    
    step = 0
    generated_tokens = []
    
    # Track prefill
    trace_system.timeline.record_event("prefill_phase", "chunked_prefill_start", stream_id=0)
    trace_system.cuda_events.record_start(f"prefill_{model_id}", "prefill")
    
    # Forward step prefill
    logits = wrapper.forward_step(input_ids, session_id=f"session_{model_id}")
    
    trace_system.cuda_events.record_end(f"prefill_{model_id}")
    trace_system.timeline.record_event("prefill_phase", "chunked_prefill_end", stream_id=0)
    trace_system.vram.record(step, actual_seq_len, {"model_id": model_id})
    trace_system.transformer.record_forward_pass(step, actual_seq_len, wrapper.num_layers)
    
    # Log layerwise activity for prefill
    for layer_idx in range(wrapper.num_layers):
        trace_system.transformer.record_layer_invocation(step, layer_idx, actual_seq_len, "prefill")
        trace_system.transformer.record_attention_call(
            step, layer_idx, 
            [1, wrapper.heads, actual_seq_len, wrapper.head_dim],
            [1, wrapper.heads, actual_seq_len, wrapper.head_dim], 
            is_sparse=False
        )
        
    next_token_id = torch.argmax(logits, dim=-1)
    generated_tokens.append(next_token_id.item())
    
    # Decode loop
    start_decode_time = time.time()
    for d_step in range(1, max_new_tokens + 1):
        step = d_step
        curr_seq_len = actual_seq_len + d_step
        
        # Capture current kernel launch timestamps
        kernel_launch_ts = time.time()
        
        # Record timeline and CUDA event start
        trace_system.timeline.record_event("decode_step", "kernel_window", stream_id=0)
        trace_system.cuda_events.record_start(f"decode_step_{d_step}", "kernel")
        
        input_token_ids = next_token_id.unsqueeze(0)
        
        # Forward pass
        logits = wrapper.forward_step(input_token_ids, session_id=f"session_{model_id}")
        
        # Record CUDA event end
        trace_system.cuda_events.record_end(f"decode_step_{d_step}")
        
        # Next token
        next_token_id = torch.argmax(logits, dim=-1)
        generated_tokens.append(next_token_id.item())
        token_text = wrapper.tokenizer.decode([next_token_id.item()])
        
        # Record raw VRAM, activity, timeline
        trace_system.vram.record(step, curr_seq_len, {"model_id": model_id})
        trace_system.transformer.record_forward_pass(step, curr_seq_len, wrapper.num_layers)
        
        for layer_idx in range(wrapper.num_layers):
            trace_system.transformer.record_layer_invocation(step, layer_idx, curr_seq_len, "sparse")
            trace_system.transformer.record_attention_call(
                step, layer_idx, 
                [1, wrapper.heads, 1, wrapper.head_dim],
                [1, wrapper.heads, curr_seq_len, wrapper.head_dim],
                is_sparse=True
            )
            
        trace_system.transformer.record_decode_token(step, next_token_id.item(), token_text)
        
        # Live output stats as strictly required:
        # - current GPU utilization
        # - current VRAM allocated
        # - current power draw
        # - current active sequence length
        # - current decode throughput
        # - current kernel launch timestamps
        gpu_util, vram_alloc, pwr_draw = get_live_gpu_status()
        elapsed = time.time() - start_decode_time
        throughput = f"{d_step / max(elapsed, 0.001):.2f} tok/s"
        
        print(
            f"GPU-Util: {gpu_util} | "
            f"VRAM: {vram_alloc} | "
            f"Power: {pwr_draw} | "
            f"Seq-Len: {curr_seq_len} | "
            f"Throughput: {throughput} | "
            f"Kernel-TS: {kernel_launch_ts:.6f}"
        )
        
        # Advance torch profiler
        trace_system.step()
        
        if next_token_id.item() == wrapper.tokenizer.eos_token_id:
            break

def run_rhd_validation():
    workspace_root = Path("d:/Codes/Projects/Differential KV")
    
    # 1. Clean/Prepare required directories
    sub_dirs = [
        "reports/stage3b/phase_41_4_6_rhd",
        "telemetry/stage3b/phase_41_4_6_rhd",
        "benchmarks/stage3b/phase_41_4_6_rhd",
        "traces/stage3b/phase_41_4_6_rhd",
        "manifests/stage3b/phase_41_4_6_rhd"
    ]
    for sd in sub_dirs:
        os.makedirs(workspace_root / sd, exist_ok=True)
        
    print("[RHD] Initializing Raw Hardware Diagnostics validation run...")
    
    # 2. Boot Raw Hardware Trace System
    trace_system = RawHardwareTraceSystem(workspace_root)
    trace_system.start_recording()
    
    # We execute sequentially the required models with targeted contexts
    try:
        # Target 1: Qwen2.5-0.5B-Instruct at 4K Context
        run_real_inference(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            context_target_tokens=4096,
            max_new_tokens=50,
            trace_system=trace_system
        )
        
        # Target 2: Qwen2.5-1.5B-Instruct at 8K Context
        run_real_inference(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            context_target_tokens=8192,
            max_new_tokens=50,
            trace_system=trace_system
        )
        
    except Exception as e:
        print(f"[RHD] Diagnostic validation failed during execution: {e}", file=sys.stderr)
    finally:
        # 3. Stop recording and flush logs
        trace_system.stop_recording()
        
    # 4. Verify integrity of captured raw logs using integrity guard
    print("[RHD] Performing raw hardware trace integrity audit...")
    guard = ScalingIntegrityGuard()
    
    trace_dir = workspace_root / "traces/stage3b/phase_41_4_6_rhd"
    telemetry_dir = workspace_root / "telemetry/stage3b/phase_41_4_6_rhd"
    
    success = guard.validate_rhd_run(trace_dir, telemetry_dir)
    
    # Create final manifest log as requested
    manifest = {
        "validation_phase": "STAGE_3B_4_6_RHD",
        "timestamp": time.time(),
        "integrity_check": "PASS" if success else "FAIL",
        "raw_artifacts": [
            str(telemetry_dir / "raw_nvidia_smi.log"),
            str(telemetry_dir / "raw_nvidia_smi_dmon.log"),
            str(telemetry_dir / "raw_torch_profiler_trace.json"),
            str(telemetry_dir / "raw_cuda_event_trace.json"),
            str(trace_dir / "raw_vram_trace.jsonl"),
            str(trace_dir / "raw_transformer_activity_trace.jsonl"),
            str(trace_dir / "raw_gpu_timeline_trace.jsonl")
        ]
    }
    
    with open(workspace_root / "manifests/stage3b/phase_41_4_6_rhd/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
        
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    run_rhd_validation()
