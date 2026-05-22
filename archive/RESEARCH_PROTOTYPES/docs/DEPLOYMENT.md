# Deployment Guide

Differential KV is designed for sparse execution efficiency.

## VRAM Optimization
Use `--sparse-mode lowrank_sparse` for RTX 40-series cards.
Use `--sparse-mode shared_basis` for data-center A100/H100 clusters.
