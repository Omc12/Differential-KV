"""
runtime/async_resonance_scheduler.py

Executes resonance pulse scheduling asynchronously to the main inference loop.
Minimizes CPU-GPU synchronization stalls.
"""

import torch
import threading
import queue
import time
from typing import Dict, Any, Callable

class AsyncResonanceScheduler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = False
        self.worker_thread = None
        
    def start(self, decision_callback: Callable):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, args=(decision_callback,))
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()

    def submit_state(self, token_idx: int, latent_features: torch.Tensor):
        """Non-blocking submission of features for resonance analysis."""
        try:
            self.input_queue.put_nowait((token_idx, latent_features.detach().cpu()))
        except queue.Full:
            pass # Skip if busy to maintain throughput

    def get_latest_decision(self) -> Optional[Dict[str, Any]]:
        """Non-blocking retrieval of the latest pulse decision."""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker(self, decision_callback: Callable):
        while self.running:
            try:
                token_idx, features = self.input_queue.get(timeout=0.1)
                # Compute resonance pulse decision in separate thread
                decision = decision_callback(token_idx, features)
                self.output_queue.put(decision)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AsyncResonanceScheduler] Error: {e}")
