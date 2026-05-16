import asyncio
import uvicorn
from runtime.lgs_resolver import LGSResolver
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

config = {
    "lgs": {
        "max_ttft_ms": 10000,
        "max_itl_ms": 500,
        "min_fairness_index": 0.8,
        "min_sparse_ratio": 0.9,
        "max_queue_wait_ms": 15000
    }
}

print("Initializing LGS Resolver and Loading Model (Qwen/Qwen2.5-0.5B-Instruct)...")
resolver = LGSResolver(config)
resolver.setup_runtime()

gateway = OpenAICompatibleAPIGateway(resolver.lgs_runtime_executor)

@gateway.app.on_event("startup")
async def startup_event():
    await gateway.start()
    print("API Gateway and Sparse Request Scheduler started.")

@gateway.app.on_event("shutdown")
async def shutdown_event():
    await gateway.stop()
    print("API Gateway and Sparse Request Scheduler stopped.")

if __name__ == "__main__":
    print("Starting Differential KV Real Serving Entrypoint...")
    uvicorn.run(gateway.app, host="0.0.0.0", port=8000)
