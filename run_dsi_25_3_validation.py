import asyncio
import logging
from typing import Dict, List, Any
from runtime.dsi_resolver import DSIResolver

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DSIValidation")

async def run_dsi_validation():
    logger.info("Starting Phase 25.3 — DSI Validation...")
    
    # 1. Initialize DSI Infrastructure
    devices = ["cuda:0", "cuda:1"]
    resolver = DSIResolver(devices)
    
    # 2. Test Autoregressive Generation
    session_id = "inference_sess_0"
    prompt = [101, 102, 103]
    resolver.start_inference_session(session_id, prompt)
    
    # 3. Simulate multi-step generation
    expected_tokens = [201, 202, 203, 204, 205]
    for i, token in enumerate(expected_tokens):
        # Alternate devices to test streaming and scheduling
        target_dev = devices[i % len(devices)]
        await resolver.run_generation_step(session_id, token, target_dev, token)
    
    logger.info("--- Test 1: Distributed Generation Continuity ---")
    tokens = resolver.orchestrator.get_session_tokens(session_id)
    logger.info(f"Final token sequence: {tokens}")
    
    logger.info("--- Test 2: Pipeline Decoding & Streaming ---")
    stability = resolver.token_streamer.get_stream_stability()
    efficiency = resolver.pipeline_decoder.get_pipeline_efficiency()
    logger.info(f"Stream stability: {stability}, Pipeline efficiency: {efficiency}")

    # 4. Collect Metrics
    metrics = resolver.get_dsi_metrics()
    
    logger.info("\n=== DSI Phase 25.3 Metrics ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    
    # 5. Final Validation Check
    success = (
        metrics["distributed_generation_integrity"] == 1.0 and
        metrics["symbolic_generation_continuity"] == 1.0 and
        metrics["token_stream_stability"] == 1.0 and
        metrics["decode_pipeline_efficiency"] >= 0.8 and
        metrics["autoregressive_replay_accuracy"] == 1.0
    )
    
    if success:
        logger.info("\nPHASE 25.3 DSI VALIDATION SUCCESSFUL")
    else:
        logger.error("\nPHASE 25.3 DSI VALIDATION FAILED")

if __name__ == "__main__":
    asyncio.run(run_dsi_validation())
