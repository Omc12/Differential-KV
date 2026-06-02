#!/usr/bin/env python3
"""
scratch/test_gateway_coherence.py

Runs the complete end-to-end API gateway integration test:
  1. Starts the FastAPI Uvicorn API gateway on port 8001.
  2. Sends concurrent chat completion requests to trigger eviction.
  3. Restores the evicted session and asks a follow-up question.
  4. Verifies coherence and kills the gateway process cleanly.
"""

import os
import sys
import subprocess
import time
import requests
import json

# Ensure ACTIVE_RUNTIME is in path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

PORT = 8001
API_URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

def main():
    print("=" * 60)
    print("  End-to-End API Gateway Integration & Coherence Test")
    print("=" * 60)

    # 1. Start the API gateway in the background
    cmd = [
        "/Users/omchimurkar1/Desktop/Differential-KV/diffkv_venv/bin/python3.14",
        "serving/openai_compatible_api_gateway.py",
        "--model", "Qwen/Qwen2.5-0.5B-Instruct",
        "--port", str(PORT),
        "--max-resident-sessions", "2",  # easy eviction
        "--serving-mode", "lightweight",
    ]
    
    print(f"Launching API gateway in background on port {PORT}...")
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.join(_parent_dir, "ACTIVE_RUNTIME"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    # Wait for the server to start
    print("Waiting 10s for model loading and server JIT warmup...")
    time.sleep(10)
    
    # Verify health
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health")
        print(f"Server health status: {r.status_code} ({r.json()})")
    except Exception as e:
        print(f"Failed to connect to gateway: {e}")
        # Print gateway stdout/stderr for debugging
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=2)
        print(f"--- SERVER STDOUT ---\n{stdout}")
        print(f"--- SERVER STDERR ---\n{stderr}")
        sys.exit(1)

    headers = {"Content-Type": "application/json"}
    
    try:
        # ── Step 1: Alice's Turn 1 ──
        print("\n1. Submitting Session A (Alice)...")
        payload_a = {
            "model": "diffkv-serving",
            "messages": [{"role": "user", "content": "Hello! I am Alice. Remember my name."}],
            "stream": False,
            "temperature": 0.0,
            "max_tokens": 32,
        }
        r_a = requests.post(API_URL, headers=headers, json=payload_a)
        res_a = r_a.json()
        ans_a = res_a["choices"][0]["message"]["content"]
        print(f"Session A response: {ans_a.strip()!r}")
        
        # Save session_id returned from non-streaming (or from standard headers/cookies)
        # Wait, the gateway returns session_id if we created one? 
        # Actually, let's create sessions explicitly via /v1/sessions first to be 100% deterministic!
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
    finally:
        print("\nStopping background API gateway server...")
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=3)
            print("Server stopped cleanly.")
        except Exception:
            proc.kill()
            print("Server force-killed.")

if __name__ == "__main__":
    main()
