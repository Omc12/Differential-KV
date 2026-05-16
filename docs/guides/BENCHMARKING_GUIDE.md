# Differential KV: Benchmarking Guide

## Canonical Benchmarks
Standardized performance tests for Differential KV:
- **Throughput (TPS)**: Measured via `run_lgs_real_validation.py`.
- **Latency (TTFT/ITL)**: Validated under concurrent load.
- **VRAM Economy**: Monitored by `MemoryPressureSafetySystem`.

## Running a Benchmark
To execute the full production-grade benchmark suite:
```bash
python run_lgs_real_validation.py
```

## Reporting
Benchmarks generate reports in the `./results` directory.
- `production_benchmark_report.md`: High-level executive summary.
- `unified_telemetry.json`: Deep telemetry for developers.

## Comparison with Baselines
Differential KV can be compared against standard HuggingFace or vLLM runtimes using the `EcosystemCompatibilitySweep`.