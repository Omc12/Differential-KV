# Rigorous Evaluation Report (Active Runtime)

| Model | Mode | Multi-Needle 4k | Multi-Needle 8k | Multi-Needle 16k | Multi-Needle 32k | Synthesis 8k | Synthesis 16k | Relational AB |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | DENSE | Y (68.7 tps) | Y (61.3 tps) | Y (36.1 tps) | Y (31.4 tps) | 23.3/100 | 0.0/100 | 4/4 |
| Qwen2.5-1.5B | ACTIVE | Y (15.6 tps) | Y (11.9 tps) | Y (9.8 tps) | Y (7.2 tps) | 10.0/100 | 6.7/100 | 4/4 |