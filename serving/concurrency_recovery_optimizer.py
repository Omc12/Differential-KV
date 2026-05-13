import torch
from typing import List, Dict
from serving.retrieval_locality_partitioner import RetrievalLocalityPartitioner
from serving.hotset_shard_allocator import HotsetShardAllocator
from serving.adaptive_concurrency_windows import AdaptiveConcurrencyWindows
from serving.local_queue_balancer import LocalQueueBalancer
from serving.retrieval_contention_predictor import RetrievalContentionPredictor
from serving.latency_pressure_scheduler import LatencyPressureScheduler
from serving.concurrency_fairness_controller import ConcurrencyFairnessController

class ConcurrencyRecoveryOptimizer:
    """
    Main optimizer for Phase 7.5C.
    Optimizes multi-user concurrency without aggressive collapse.
    """
    def __init__(self, num_zones: int = 4):
        self.locality_partitioner = RetrievalLocalityPartitioner(num_zones)
        self.shard_allocator = HotsetShardAllocator()
        self.concurrency_window = AdaptiveConcurrencyWindows()
        self.balancer = LocalQueueBalancer(num_zones)
        self.contention_pred = RetrievalContentionPredictor()
        self.scheduler = LatencyPressureScheduler()
        self.fairness_ctrl = ConcurrencyFairnessController()

    def submit_request(self, request: dict, seq_len: int):
        """Pre-processes and enqueues a request."""
        # 1. Assign zone affinity
        request = self.locality_partitioner.partition_request(request, seq_len)
        # 2. Record arrival for scheduling
        import time
        request["arrival_time"] = time.perf_counter()
        self.balancer.enqueue(request)

    def process_step(self, p95_latency: float, all_active_indices: torch.Tensor) -> List[dict]:
        """
        Executes one serving optimization step:
        1. Update concurrency window
        2. Predict contention and shard hotsets
        3. Dequeue and schedule a batch
        4. Apply fairness constraints
        """
        # 1. Update window
        limit = self.concurrency_window.update_window(p95_latency)
        
        # 2. Manage contention
        self.contention_pred.update_usage(all_active_indices)
        hot_indices = self.contention_pred.predict_contention()
        self.shard_allocator.allocate_shards(hot_indices)
        
        # 3. Pull balanced batch
        raw_batch = self.balancer.dequeue_batch(limit)
        
        # 4. Schedule by pressure/latency
        scheduled_batch = self.scheduler.rank_requests(raw_batch)
        
        return scheduled_batch

    def record_completion(self, user_id: str, tokens_processed: int):
        self.fairness_ctrl.record_usage(user_id, tokens_processed)
