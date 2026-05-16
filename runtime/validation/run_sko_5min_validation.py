"""
SKO 5-Minute REAL Validation Script (Hardened v3)

Orchestrates a REAL 5-minute sustained serving load using TRUE live streaming.
REMOVED: all replay-based ITL, intermediate-token TPS, and synthetic word splitting.
ADDED: Real tokenizer counting after reconstruction, request-id summaries, and decode overlap tracking.
"""
import requests
import time
import os
import json
import torch
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

class SKORealValidator:
    def __init__(self, duration=300): # 5 minutes
        self.duration = duration
        self.url = "http://localhost:8000/v1/chat/completions"
        self.telemetry_dir = "telemetry/stage2/phase_38_6_sko/"
        self.traces_dir = "traces/stage2/phase_38_6_sko/"
        os.makedirs(self.telemetry_dir, exist_ok=True)
        os.makedirs(self.traces_dir, exist_ok=True)
        
        self.dmon_log = os.path.join(self.telemetry_dir, "raw_nvidia_smi_dmon.log")
        self.request_trace = os.path.join(self.traces_dir, "request_trace.jsonl")
        self.error_log = os.path.join(self.traces_dir, "validation_errors.jsonl")
        self.concurrency_trace = os.path.join(self.traces_dir, "concurrency_trace.jsonl")
        self.request_summary = os.path.join(self.traces_dir, "final_request_summary.jsonl")
        
        # Truncate previous logs
        for f in [self.dmon_log, self.request_trace, self.error_log, self.concurrency_trace, self.request_summary]:
            with open(f, "w") as _:
                pass

        self.is_running = True
        self.active_concurrency = 0
        self.decode_overlap_count = 0
        self.concurrency_lock = threading.Lock()
        
        print("[*] Loading real Qwen tokenizer for final reconstruction counting...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    def start_telemetry(self):
        print(f"[*] Starting REAL telemetry capture: {self.dmon_log}")
        import subprocess
        cmd = f"nvidia-smi dmon -s pmu > {self.dmon_log}"
        return subprocess.Popen(cmd, shell=True)

    def log_concurrency(self):
        start_time = time.time()
        while self.is_running and (time.time() - start_time < self.duration):
            with self.concurrency_lock:
                entry = {
                    "timestamp": time.time(),
                    "active_requests": self.active_concurrency,
                    "decode_overlap": self.decode_overlap_count
                }
            with open(self.concurrency_trace, "a") as f:
                f.write(json.dumps(entry) + "\n")
            time.sleep(0.5)

    def run_session(self, session_id):
        """
        Performs TRUE Live Autoregressive Streaming Requests.
        """
        start_wall = time.time()
        while time.time() - start_wall < self.duration:
            with self.concurrency_lock:
                self.active_concurrency += 1
            
            request_id = f"req-{uuid.uuid4()}"
            payload = {
                "model": "diffkv-qwen2.5-7b",
                "messages": [{"role": "user", "content": "Explain the physical implications of sparse kernel optimization."}],
                "stream": True,
                "max_tokens": 100
            }
            
            request_start = time.time()
            full_text = ""
            status = "failed"
            chunks_received = 0
            ttft = None
            server_timings = {}
            
            try:
                response = requests.post(self.url, json=payload, stream=True, timeout=300)
                
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith("data: "):
                            data_str = line_text[6:]
                            if data_str == "[DONE]":
                                status = "completed"
                                break
                            
                            try:
                                data = json.loads(data_str)
                                chunks_received += 1
                                
                                # Check for server timings (sent in chunks or final)
                                if "usage" in data:
                                    server_timings = data["usage"].get("server_timings", {})
                                
                                # Update decode overlap (if server says we are decoding)
                                if "server_timings" in data and "decode_complete_ts" in data["server_timings"]:
                                    # Simple heuristic: if we received a token chunk, we are decoding
                                    with self.concurrency_lock:
                                        self.decode_overlap_count = max(0, self.active_concurrency - 1) # Approximation

                                token_text = ""
                                if data.get('choices'):
                                    token_text = data['choices'][0]['delta'].get('content', '')
                                
                                if token_text:
                                    if ttft is None:
                                        ttft = (time.time() - request_start) * 1000
                                        print(f"\n[Session {session_id}] LIVE TOKEN RECEIVED (TTFT: {ttft:.2f}ms)")
                                    
                                    full_text += token_text
                                    print(token_text, end="", flush=True)
                                    
                                    # Periodic trace log (optional, only for raw timeline)
                                    with open(self.request_trace, "a") as f:
                                        trace_entry = {
                                            "timestamp": time.time(),
                                            "request_id": request_id,
                                            "chunk_index": chunks_received,
                                            "server_decode_ts": data.get("server_timings", {}).get("decode_complete_ts")
                                        }
                                        f.write(json.dumps(trace_entry) + "\n")
                            except:
                                continue
                                
            except Exception as e:
                status = "timeout" if "timeout" in str(e).lower() else "failed"
                error_entry = {"timestamp": time.time(), "request_id": request_id, "error": str(e)}
                with open(self.error_log, "a") as f:
                    f.write(json.dumps(error_entry) + "\n")
                print(f"\n[!] Request {request_id} Error: {e}")
                time.sleep(2)
            finally:
                request_end = time.time()
                # FINAL TOKEN COUNTING (RECONSTRUCTED)
                final_token_count = len(self.tokenizer.encode(full_text, add_special_tokens=False))
                
                summary = {
                    "request_id": request_id,
                    "session_id": session_id,
                    "total_real_tokens": final_token_count,
                    "total_chunks": chunks_received,
                    "request_duration_ms": (request_end - request_start) * 1000,
                    "ttft_ms": ttft,
                    "status": status,
                    "server_timings": server_timings
                }
                with open(self.request_summary, "a") as f:
                    f.write(json.dumps(summary) + "\n")
                
                print(f"\n[Session {session_id}] Request Summary: {final_token_count} tokens, {status}")
                
                with self.concurrency_lock:
                    self.active_concurrency -= 1

    def run(self):
        print("=====================================================")
        print("SKO 5-MINUTE REAL INFERENCE VALIDATION (HARDENED v3)")
        print("Target: http://localhost:8000/v1/chat/completions")
        print("Mode: TRUE LIVE DECODE STREAMING")
        print("=====================================================")
        
        # LIVE RUNTIME VERIFICATION
        print("[*] Performing LIVE Runtime Verification...")
        try:
            runtime_info = requests.get("http://localhost:8000/v1/runtime_info", timeout=5).json()
            if runtime_info["streaming_mode"] != "live_autoregressive":
                print(f"[!] VERIFICATION FAIL: Server is NOT in live streaming mode: {runtime_info}")
                sys.exit(1)
            print(f"[+] VERIFICATION SUCCESS: {runtime_info['model']} in {runtime_info['streaming_mode']} mode.")
        except Exception as e:
            print(f"[!] VERIFICATION ERROR: Could not connect to /v1/runtime_info: {e}")
            sys.exit(1)

        telemetry_proc = self.start_telemetry()
        
        try:
            import uuid
            with ThreadPoolExecutor(max_workers=9) as executor:
                executor.submit(self.log_concurrency)
                for i in range(8):
                    executor.submit(self.run_session, i)
                
                start_time = time.time()
                while time.time() - start_time < self.duration:
                    elapsed = time.time() - start_time
                    print(f"\r[REAL-TIME] Elapsed: {elapsed:.1f}s | Active: {self.active_concurrency} | Decodes: {self.decode_overlap_count}", end="")
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.is_running = False
            if telemetry_proc:
                import subprocess
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(telemetry_proc.pid)], shell=True)
            print("\n\n[*] REAL Validation Complete. RAW logs only produced.")

if __name__ == "__main__":
    validator = SKORealValidator(duration=300)
    validator.run()
