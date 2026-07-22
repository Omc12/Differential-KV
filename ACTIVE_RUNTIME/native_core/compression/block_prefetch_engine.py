"""
BlockPrefetchEngine: Step-ahead block prefetch daemon.

This module provides background asynchronous H2D restoration of tiered CPU
slots so they are warm for subsequent decoding steps. It utilizes a background
thread and thread-safe queue to avoid blocking the main execution path.
"""

import os
import threading
import queue
import time
from typing import List, Optional

import torch


class BlockPrefetchEngine:
    def __init__(self, tiered_store, device: str, lookahead: int = 1):
        self.tiered_store = tiered_store
        self.device = str(device)
        self.lookahead = lookahead
        
        self._queue = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
    def is_enabled(self) -> bool:
        if self.tiered_store is None:
            return False
        return self.device.startswith("cuda") or self.device.startswith("mps")
        
    def start(self):
        if not self.is_enabled():
            return
            
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._shutdown_event.clear()
        self._thread = threading.Thread(
            target=self._worker, 
            daemon=True, 
            name="BlockPrefetchEngine"
        )
        self._thread.start()
        
    def stop(self):
        if self._thread is not None:
            self._shutdown_event.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            
    def submit(self, session_id: str, routed_slots: List[int]) -> bool:
        if not self.is_enabled():
            return False
            
        try:
            self._queue.put_nowait((session_id, routed_slots))
            return True
        except queue.Full:
            return False
            
    def _worker(self):
        while not self._shutdown_event.is_set():
            try:
                # Use a small timeout to allow checking shutdown_event
                job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            session_id, routed_slots = job
            try:
                # Execute without GIL where possible during torch ops
                with torch.no_grad():
                    cold_slots = [
                        s for s in routed_slots
                        if self.tiered_store.get_tier(s) == 'CPU'
                    ]
                    
                    if cold_slots:
                        warming = self.tiered_store.ensure_warm(cold_slots)
                        
                        if os.environ.get("DIFFKV_TELEMETRY") == "1":
                            print(f"[Prefetch] session={session_id} warmed {len(warming)} slots")
            except Exception as e:
                # Log or handle worker exceptions gracefully
                pass
            finally:
                self._queue.task_done()
                
    def sync_pending(self, session_id: str, routed_slots: List[int]):
        if not self.is_enabled():
            return
            
        with torch.no_grad():
            self.tiered_store.sync_warming_slots(routed_slots)
