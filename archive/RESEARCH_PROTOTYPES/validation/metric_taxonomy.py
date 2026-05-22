"""
validation/metric_taxonomy.py

Strict definitions and constants for Differential KV performance metrics.
Ensures scientific credibility by separating kernel, retrieval, and serving throughput.
"""

from enum import Enum

class MetricClass(Enum):
    KERNEL_THROUGHPUT = "Kernel Throughput"
    RETRIEVAL_OPS = "Sparse Retrieval Ops/sec"
    SERVING_TPS = "End-to-End Serving TPS"
    TOKEN_THROUGHPUT = "Effective Token Throughput"
    SYNC_THROUGHPUT = "Distributed Synchronization Throughput"
    MICROBENCHMARK = "Synthetic Microbenchmark"
    SIMULATED = "Simulated Result"
    THEORETICAL = "Theoretical Estimate"

class TruthStatus(Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    SIMULATED = "SIMULATED"

# Mandatory Taxonomy Requirements
TAXONOMY_MAP = {
    "tps": MetricClass.SERVING_TPS,
    "kernel_time": MetricClass.KERNEL_THROUGHPUT,
    "retrieval_latency": MetricClass.RETRIEVAL_OPS,
    "sync_overhead": MetricClass.SYNC_THROUGHPUT,
    "theoretical_bandwidth": MetricClass.THEORETICAL,
    "simulated_scaling": MetricClass.SIMULATED
}
