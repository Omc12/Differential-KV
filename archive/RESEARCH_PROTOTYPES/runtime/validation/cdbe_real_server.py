import asyncio
import time
import torch
import uvicorn
import logging
import sys
import os
import json
from typing import Dict, List, Any
from fastapi import FastAPI

from runtime.cdbe_resolver import CDBEResolver
from runtime.hf_dkv_wrapper import DKVHFWrapper
from decode_pipeline_fusion_engine import DecodePipelineFusionEngine
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CDBEServer")

model_id = "Qwen/Qwen2.5-7B-Instruct"
resolver = None
gateway = None

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global resolver, gateway
    print(f"[*] CDBE STAGE 2: Loading REAL 7B Model -> {model_id}")
    
    # Initialize Wrapper
    wrapper = DKVHFWrapper(model_id, {
        "mode": "lowrank_sparse", 
        "block_size": 64, 
        "rank": 16
    })
    
    # Initialize Fusion Engine
    fusion_engine = DecodePipelineFusionEngine(wrapper)
    
    # Initialize Resolver
    resolver = CDBEResolver(wrapper, fusion_engine)
    await resolver.start()
    
    # Shim for gateway
    async def shim_stream_executor(session_ids: List[str], payloads: List[Dict[str, Any]]):
        session_id = session_ids[0]
        payload = payloads[0]
        payload["session_id"] = session_id
        async for chunk in resolver.execute_stream(payload):
            yield {
                "step": 0,
                "chunks": [{
                    "session_id": session_id,
                    "token_text": chunk["token_text"],
                    "decode_complete_ts": chunk["decode_complete_ts"],
                    "is_final": chunk["is_final"]
                }]
            }
    
    resolver.lgs_runtime_stream_executor = shim_stream_executor
    
    # Initialize Gateway routes on our app
    gateway = OpenAICompatibleAPIGateway(resolver)
    app.include_router(gateway.app.router)
    
    # Start telemetry loop
    asyncio.create_task(telemetry_loop())
    print("[*] CDBE Resolver Infrastructure ONLINE.")

async def telemetry_loop():
    print("\n" + "="*60)
    print("CDBE REAL-TIME RUNTIME MONITOR")
    print("="*60)
    
    while True:
        try:
            if resolver and resolver.worker:
                worker_stats = resolver.worker.get_occupancy_stats()
                scheduler_stats = resolver.scheduler.get_scheduler_status()
                overlap = len(resolver.telemetry.active_sessions)
                
                print(
                    f"[LIVE] Req: {worker_stats['active_sessions']:2d} | "
                    f"Batch: {worker_stats['last_batch_size']:3d} | "
                    f"Queue: {scheduler_stats['queue_depth']:3d} | "
                    f"Overlap: {overlap:2d} | "
                    f"Steps: {worker_stats['total_steps']:6d}"
                )
                
                # Persist traces
                ts = time.time()
                os.makedirs("traces/stage2/phase_38_7_cdbe", exist_ok=True)
                with open("traces/stage2/phase_38_7_cdbe/live_request_trace.jsonl", "a") as f:
                    f.write(json.dumps({"ts": ts, "active_sessions": worker_stats['active_sessions']}) + "\n")
                with open("traces/stage2/phase_38_7_cdbe/live_batch_trace.jsonl", "a") as f:
                    f.write(json.dumps({"ts": ts, "batch_size": worker_stats['last_batch_size']}) + "\n")
        except Exception as e:
            pass
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
