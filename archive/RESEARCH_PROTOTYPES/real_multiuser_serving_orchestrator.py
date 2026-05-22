import asyncio
import time
import uuid
import random
from typing import List, Dict, Any, Optional
from serving.openai_compatible_api_gateway import OpenAICompatibleAPIGateway, ChatCompletionRequest
from serving.production_session_manager import ProductionSessionManager

class RealMultiUserServingOrchestrator:
    """
    PSR System 1: Real Multi-User Serving Orchestrator.
    Manages concurrent user sessions, request arrival patterns, and queue contention.
    """
    def __init__(self, api_gateway: OpenAICompatibleAPIGateway):
        self.api_gateway = api_gateway
        self.session_manager = api_gateway.session_manager
        self.active_users: Dict[str, asyncio.Task] = {}
        self.request_history: List[Dict[str, Any]] = []
        self.is_running = False

    async def simulate_user_session(self, user_id: str, workload_type: str, delay_mean: float = 1.0):
        """Simulates a single user's interaction with the serving system."""
        while self.is_running:
            # Realistic request arrival (Poisson-like delay)
            await asyncio.sleep(random.expovariate(1.0 / delay_mean))  # Faster for HSM
            
            # Prepare realistic workload
            if workload_type == "short_chat":
                max_tokens = random.randint(20, 100)
                prompt = "Hello, tell me a short joke."
            elif workload_type == "long_context":
                max_tokens = random.randint(500, 2000)
                prompt = "Summarize this long document: " + "word " * 2000
            elif workload_type == "code_gen":
                max_tokens = random.randint(100, 500)
                prompt = "Write a python function to sort a list of dictionaries by a key."
            else:
                max_tokens = 100
                prompt = "Standard request."

            request = ChatCompletionRequest(
                model="qwen2.5-7b-instruct",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=max_tokens,
                session_id=user_id
            )

            start_time = time.time()
            try:
                # Direct call to the gateway's logic to capture serving overhead
                response_gen = self.api_gateway._stream_generator(
                    user_id, f"req-{uuid.uuid4()}", int(start_time), "qwen2.5-7b-instruct", 
                    {"prompt": prompt, "max_tokens": max_tokens}
                )
                
                # Consume stream to simulate real user behavior
                token_count = 0
                first_token_time = None
                async for chunk in response_gen:
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1
                
                end_time = time.time()
                self.request_history.append({
                    "user_id": user_id,
                    "workload": workload_type,
                    "ttft": first_token_time - start_time if first_token_time else 0,
                    "total_time": end_time - start_time,
                    "tokens": token_count,
                    "tps": token_count / (end_time - first_token_time) if first_token_time and end_time > first_token_time else 0
                })
            except Exception as e:
                print(f"User {user_id} request failed: {e}")

    async def start_sustained_load(self, concurrency: int, duration_secs: int):
        """Starts a sustained multi-user load test."""
        self.is_running = True
        workload_types = ["short_chat", "long_context", "code_gen", "summarization"]
        
        tasks = []
        for i in range(concurrency):
            user_id = f"user-{i}"
            workload = random.choice(workload_types)
            tasks.append(asyncio.create_task(self.simulate_user_session(user_id, workload)))
        
        await asyncio.sleep(duration_secs)
        self.is_running = False
        await asyncio.gather(*tasks, return_exceptions=True)

    def get_orchestration_stats(self):
        return {
            "total_requests": len(self.request_history),
            "concurrency": len(self.active_users),
            "avg_ttft": sum(r["ttft"] for r in self.request_history) / len(self.request_history) if self.request_history else 0,
            "avg_tps": sum(r["tps"] for r in self.request_history) / len(self.request_history) if self.request_history else 0
        }
