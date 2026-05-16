import asyncio
import time
import torch
import os
import json
import logging
import sys
from typing import List, Dict, Any

from transformers import BitsAndBytesConfig
from runtime.cdbe_resolver import CDBEResolver
from runtime.hf_diffkv_wrapper import DiffKVHFWrapper
from decode_pipeline_fusion_engine import DecodePipelineFusionEngine

# DQO Imports
from runtime.sustained_decode_continuity_monitor import SustainedDecodeContinuityMonitor
from runtime.live_throughput_variance_tracker import LiveThroughputVarianceTracker
from runtime.batch_efficiency_instrumentation import BatchEfficiencyInstrumentation
from runtime.adaptive_decode_window_controller import AdaptiveDecodeWindowController
from runtime.queue_pressure_stabilizer import QueuePressureStabilizer

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] DQO: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DQO_Validation")

async def run_dqo_validation():
    """
    STAGE 2 DQO: Decode Quality Optimization Validation.
    Focuses on sustained decode efficiency and quality under real pressure.
    """
    logger.info("=== DQO REAL RUNTIME VALIDATION STARTING ===")
    
    # 1. Environment & Model Setup
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use 4-bit quantization to fit 7B on 12GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    logger.info(f"Target Model: {model_id} on {device} (4-bit NF4)")
    
    try:
        wrapper = DiffKVHFWrapper(
            model_id, 
            {"mode": "lowrank_sparse", "block_size": 64, "rank": 16},
            quantization_config=bnb_config
        )
        fusion_engine = DecodePipelineFusionEngine(wrapper)
        resolver = CDBEResolver(wrapper, fusion_engine)
        
        # 2. DQO Component Initialization
        continuity_monitor = SustainedDecodeContinuityMonitor()
        throughput_tracker = LiveThroughputVarianceTracker()
        efficiency_instrumentation = BatchEfficiencyInstrumentation()
        window_controller = AdaptiveDecodeWindowController(max_window_size=32) # Constraint for 12GB
        pressure_stabilizer = QueuePressureStabilizer()
        
        # 3. Injection into Runtime
        resolver.worker.set_dqo_instrumentation(
            continuity_monitor,
            throughput_tracker,
            efficiency_instrumentation
        )
        
        await resolver.start()
        logger.info("DQO Infrastructure ONLINE & Instrumented.")
        
    except Exception as e:
        logger.error(f"Failed to initialize DQO Runtime: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Define Workload
    # 8-16 Concurrent sessions, REAL prompts
    concurrency = 12 
    duration_limit_sec = 300 # 5 minutes
    
    # Real prompts (Long context)
    real_prompts = [
        "Explain the principles of general relativity in the context of gravitational wave detection.",
        "Write a detailed technical specification for a block-sparse attention kernel in Triton.",
        "Discuss the evolution of large language models from BERT to Qwen2.5.",
        "Provide a comprehensive overview of quantum computing algorithms for optimization.",
        "Analyze the impact of KV cache compression on long-context retrieval performance."
    ]
    
    async def run_session(session_id: str):
        prompt = real_prompts[hash(session_id) % len(real_prompts)]
        payload = {
            "session_id": session_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024 # Long streaming
        }
        
        token_count = 0
        try:
            async for chunk in resolver.execute_stream(payload):
                token_count += chunk["token_count"]
                # Simulate user reading time or just keep pushing
        except Exception as e:
            logger.error(f"Session {session_id} error: {e}")
        return token_count

    # 5. Live Execution Loop with Observability
    logger.info(f"Launching {concurrency} concurrent DQO sessions...")
    start_time = time.time()
    tasks = [asyncio.create_task(run_session(f"dqo-{i}")) for i in range(concurrency)]
    
    try:
        while time.time() - start_time < duration_limit_sec:
            if all(t.done() for t in tasks):
                break
                
            elapsed = time.time() - start_time
            worker_stats = resolver.worker.get_occupancy_stats()
            t_metrics = throughput_tracker.get_variance_metrics()
            c_metrics = continuity_monitor.get_metrics()
            e_metrics = efficiency_instrumentation.get_efficiency_metrics()
            
            # Update adaptive window based on pressure
            queue_depth = e_metrics.get("avg_queue_depth", 0)
            overlap_count = worker_stats.get("active_sessions", 0)
            
            # Stabilize pressure signal
            smoothed_pressure = pressure_stabilizer.update_pressure(int(queue_depth))
            
            new_window = window_controller.adjust_window(
                int(smoothed_pressure), 
                int(overlap_count),
                0.0, # Placeholder for latency
                t_metrics.get("current_tps", 0.0)
            )
            resolver.worker.max_batch_size = new_window
            
            # LIVE OBSERVABILITY PRINT
            print(f"\r[DQO LIVE] {int(elapsed)}s | Sessions: {worker_stats['active_sessions']} | "
                  f"Batch: {worker_stats['last_batch_size']} | TPS: {t_metrics['current_tps']:.1f} | "
                  f"Continuity: {c_metrics['continuity_score']:.3f} | Var: {t_metrics['tps_variance']:.1f} | "
                  f"Queue: {int(queue_depth)}", end="", flush=True)
            
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user.")
    
    print("\n")
    logger.info("DQO Validation duration reached or tasks complete. Cleaning up...")
    
    # 6. Final Data Collection
    end_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_tokens = sum([r for r in results if isinstance(r, int)])
    
    summary = {
        "duration_sec": end_time - start_time,
        "total_tokens": total_tokens,
        "avg_tps": total_tokens / (end_time - start_time),
        "final_metrics": {
            "throughput": throughput_tracker.get_variance_metrics(),
            "continuity": continuity_monitor.get_metrics(),
            "efficiency": efficiency_instrumentation.get_efficiency_metrics(),
            "window_config": window_controller.get_config()
        }
    }
    
    with open("reports/stage2/phase_38_8_dqo/dqo_validation_report.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    logger.info("=== DQO VALIDATION COMPLETE ===")
    logger.info(f"Final Continuity Score: {summary['final_metrics']['continuity']['continuity_score']:.3f}")
    logger.info(f"Avg Throughput: {summary['avg_tps']:.2f} tokens/sec")
    logger.info(f"Report saved to reports/stage2/phase_38_8_dqo/dqo_validation_report.json")

    await resolver.stop()

if __name__ == "__main__":
    asyncio.run(run_dqo_validation())
