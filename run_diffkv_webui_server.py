import os
import sys
import asyncio
import uvicorn
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from runtime.unified_runtime_packaging_layer import UnifiedRuntimePackagingLayer
from runtime.production_session_lifecycle_manager import ProductionSessionLifecycleManager
from runtime.browser_failure_recovery_layer import BrowserFailureRecoveryLayer
from runtime.openai_compatibility_stability_layer import OpenAICompatibilityStabilityLayer
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DiffKV-Production")

async def main():
    print("\n" + "="*60)
    print("DIFFERENTIAL KV — PRODUCTION RUNTIME ENTRYPOINT (STAGES 1-3)")
    print("="*60 + "\n")

    workspace_root = Path("d:/Codes/Projects/Differential KV")
    
    # 1. Boot Unified Runtime (Packaging Layer)
    # This initializes Stage 1 (Semantic Governance) and Stage 2 (CDBE Engine)
    runtime = UnifiedRuntimePackagingLayer(workspace_root)
    if not runtime.boot(mode="production"):
        logger.error("CRITICAL: Runtime boot failed. Check hardware/environment.")
        return

    # 2. Initialize Operational Systems
    session_manager = ProductionSessionLifecycleManager()
    recovery_layer = BrowserFailureRecoveryLayer(session_manager)
    stability_layer = OpenAICompatibilityStabilityLayer(session_manager)

    # 3. Setup API Gateway
    # We use the existing gateway but bridge it to the new runtime components
    # Note: We pass the runtime object which contains the executors
    gateway = OpenAICompatibleAPIGateway(runtime)
    
    # Override session manager in gateway to use the new lifecycle manager
    gateway.session_manager = session_manager

    @gateway.app.on_event("startup")
    async def startup_event():
        # Here we would start the scheduler if it's not already handled by CDBE
        logger.info("Differential KV Production Gateway started.")
        logger.info("OpenAI-Compatible endpoint active at: http://localhost:8000/v1")
        logger.info("Target Model ID: diffkv-qwen2.5-7b")

    @gateway.app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Differential KV Production Gateway...")

    # 4. Launch Server
    config = uvicorn.Config(gateway.app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    print("\n[READY] Differential KV is now usable with WebUI (e.g. Open WebUI, LibreChat).")
    print("Configure your WebUI to point to: http://localhost:8000/v1")
    print("Select model: diffkv-qwen2.5-7b\n")

    await server.serve()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
