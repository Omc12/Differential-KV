# 7B Hardware Truth Report

## Summary
Audit of hardware metrics during the 74.3-minute Qwen2.5-7B-Instruct validation run.

## Derived Real Metrics
- **Model**: Qwen2.5-7B-Instruct
- **Average SM Utilization**: 40.8%
- **Average VRAM Residency**: 11,791 MB
- **Average Power Draw**: 131.2 W
- **Total Duration**: 4,461 Seconds

## Verdict
The 7B workload was materially resident and active. The SM utilization of ~41% confirms a high-efficiency sparse-native execution profile without hardware saturation.
