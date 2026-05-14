import time
import psutil
import torch
import random
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import Dict

app = FastAPI(title="DiffKV Live Dashboard")

# Mock metrics for the dashboard - in production, this would read from the runtime's telemetry
def get_live_metrics():
    vram_used = 0
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
    
    return {
        "timestamp": time.time(),
        "tps": random.uniform(15.0, 25.0), # Simulated for the dashboard view
        "vram_gb": vram_used,
        "paging_activity": random.uniform(0.1, 0.5),
        "hit_rate": random.uniform(0.95, 0.99),
        "cpu_percent": psutil.cpu_percent(),
        "ram_gb": psutil.virtual_memory().used / (1024**3)
    }

@app.get("/metrics")
async def metrics():
    return get_live_metrics()

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("telemetry/dashboard.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
