#!/usr/bin/env python3
"""
benchmarks/tool_calling_agent_eval.py — Agent Memory Benchmark v2.0

Comprehensive evaluation of KV-cache retention under realistic, tool-heavy multi-turn agent workflows:
  1. Interleaved Multi-Tool Traces (SQL, Vector Search, K8s Scaling, Payments, Auth, Billing).
  2. Irregular Jagged Conversational Flow (User, Assistant Reasoning, Tool Calls, Tool Responses, Follow-ups).
  3. High-Collision Distractors (multiple similar tool calls with varying parameters).
  4. Tool Outputs, Chained Tool Passes, & Evolving Working Memory State Tracking (Quota deductions, Replica updates).
  5. High-Entropy Diverse Agent Domain Library (Git, CI/CD, K8s, Redis, Postgres, Kafka, DNS, IAM, Alertmanager).
  6. Strict Normalized Exact Value Matching (strips formatting, verifies exact parameter/value recovery).

Model-Agnostic: run on any MLX or HuggingFace model via --model.
"""
import os
import sys
import time
import argparse
import json
import re
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ACTIVE = os.path.join(REPO, "ACTIVE_RUNTIME")
sys.path.insert(0, ACTIVE)

# ==============================================================================
# 1. DIVERSE AGENT DOMAIN LIBRARY (High-Entropy Non-Repetitive Filler)
# ==============================================================================
DIVERSE_AGENT_DOMAINS = [
    [
        {"role": "user", "content": "The staging Kubernetes cluster is throwing CrashLoopBackOff errors on pod ingress-api-77d9."},
        {"role": "assistant", "content": '{"name": "kubectl_logs", "arguments": {"namespace": "staging", "pod": "ingress-api-77d9", "tail": 100}}'},
        {"role": "tool", "content": '{"status": "error", "exit_code": 137, "reason": "OOMKilled", "memory_limit_mb": 512, "peak_usage_mb": 524}'},
        {"role": "assistant", "content": "The pod was OOMKilled exceeding its 512MB limit. I will patch the deployment manifest to increase memory allocation to 1024MB."}
    ],
    [
        {"role": "user", "content": "Check Pull Request #482 on GitHub repository enterprise/core-engine for CI build failures."},
        {"role": "assistant", "content": '{"name": "gh_pr_status", "arguments": {"repo": "enterprise/core-engine", "pr_number": 482}}'},
        {"role": "tool", "content": '{"pr_title": "feat: add vector index caching", "ci_status": "failure", "failed_job": "test-integration-cuda", "commit_sha": "a7f9201b"}'},
        {"role": "assistant", "content": "PR #482 failed on test-integration-cuda at commit a7f9201b due to a missing CUDA driver header."}
    ],
    [
        {"role": "user", "content": "We need to run database migration 014_add_user_tenant_index.sql on Postgres primary."},
        {"role": "assistant", "content": '{"name": "pg_migrate_check", "arguments": {"db_name": "production_main", "migration_id": "014_add_user_tenant_index"}}'},
        {"role": "tool", "content": '{"dry_run": true, "table": "user_profiles", "estimated_lock_sec": 0.4, "status": "safe_to_apply"}'},
        {"role": "assistant", "content": "Dry run confirmed. The index creation on user_profiles will hold a exclusive lock for under 0.4 seconds."}
    ],
    [
        {"role": "user", "content": "Update the Cloudflare DNS record for api.internal-analytics.io to point to new load balancer LB-991."},
        {"role": "assistant", "content": '{"name": "cloudflare_update_dns", "arguments": {"zone": "internal-analytics.io", "record_type": "A", "name": "api", "content": "192.0.2.148", "ttl": 300}}'},
        {"role": "tool", "content": '{"success": true, "modified_record_id": "dns_rec_88412", "zone_id": "zf_990141"}'},
        {"role": "assistant", "content": "Updated A record api.internal-analytics.io to 192.0.2.148 with TTL 300 seconds."}
    ],
    [
        {"role": "user", "content": "Audit S3 bucket security settings for customer-backups-archive-us-east."},
        {"role": "assistant", "content": '{"name": "aws_s3_audit", "arguments": {"bucket": "customer-backups-archive-us-east"}}'},
        {"role": "tool", "content": '{"public_access_block": true, "encryption": "AES256", "versioning": "Enabled", "lifecycle_policy": "GlacierAfter90Days"}'},
        {"role": "assistant", "content": "S3 bucket is secure with public access blocked, AES256 encryption enabled, and lifecycle transition to Glacier after 90 days."}
    ],
    [
        {"role": "user", "content": "Inspect Redis cache memory fragmentation and eviction metrics on node redis-cache-01."},
        {"role": "assistant", "content": '{"name": "redis_info_memory", "arguments": {"node_host": "10.0.4.12", "port": 6379}}'},
        {"role": "tool", "content": '{"used_memory_human": "14.2G", "eviction_policy": "volatile-lru", "evicted_keys_total": 489102, "mem_fragmentation_ratio": 1.18}'},
        {"role": "assistant", "content": "Redis memory is at 14.2GB with healthy fragmentation ratio 1.18 using volatile-lru eviction."}
    ],
    [
        {"role": "user", "content": "Check Kafka consumer group lag for payment-processing-group on topic transaction-events."},
        {"role": "assistant", "content": '{"name": "kafka_consumer_lag", "arguments": {"group": "payment-processing-group", "topic": "transaction-events"}}'},
        {"role": "tool", "content": '{"total_lag": 1420, "partition_lags": [{"partition": 0, "lag": 420}, {"partition": 1, "lag": 1000}]}'},
        {"role": "assistant", "content": "Kafka consumer group has a total lag of 1420 messages concentrated on partition 1."}
    ],
    [
        {"role": "user", "content": "Verify Stripe webhook signature endpoint /api/v1/webhooks/stripe for missing events."},
        {"role": "assistant", "content": '{"name": "stripe_verify_webhook", "arguments": {"endpoint": "/api/v1/webhooks/stripe", "time_window_hours": 24}}'},
        {"role": "tool", "content": '{"events_received": 14290, "failed_signatures": 0, "status": "healthy"}'},
        {"role": "assistant", "content": "Stripe webhook endpoint processed 14,290 events in the last 24 hours with 0 signature failures."}
    ]
]

# ==============================================================================
# 2. STATE-TRACKING & HIGH-COLLISION AGENT SCENARIOS
# ==============================================================================
AGENT_EVAL_TASKS = [
    {
        "id": "task_1_high_collision_sql_tenant",
        "description": "High-Collision Interference: Execute multiple similar SQL queries for different tenants; ask for limit parameter of TENANT-1102-Y",
        "interactive_turns": [
            {
                "user": "Execute database query for customer TENANT-9942-X to fetch account details.",
                "tool_call": '{"name": "execute_sql", "arguments": {"query": "SELECT user_id, tier FROM user_profiles WHERE tenant_id = \'TENANT-9942-X\'", "limit": 100}}',
                "tool_response": '{"status": 200, "rows": [{"user_id": 98412, "tier": "enterprise"}]}'
            },
            {
                "user": "Now execute database query for customer TENANT-1102-Y to inspect quota bounds.",
                "tool_call": '{"name": "execute_sql", "arguments": {"query": "SELECT user_id, quota FROM user_profiles WHERE tenant_id = \'TENANT-1102-Y\'", "limit": 250}}',
                "tool_response": '{"status": 200, "rows": [{"user_id": 44102, "quota": 300000}]}'
            },
            {
                "user": "Also execute database query for customer TENANT-7731-Z for regional compliance.",
                "tool_call": '{"name": "execute_sql", "arguments": {"query": "SELECT user_id, region FROM user_profiles WHERE tenant_id = \'TENANT-7731-Z\'", "limit": 500}}',
                "tool_response": '{"status": 200, "rows": [{"user_id": 11094, "region": "EU-West"}]}'
            }
        ],
        "query": "What was the exact limit parameter integer specified in the execute_sql query for customer TENANT-1102-Y?",
        "target": "250"
    },
    {
        "id": "task_2_high_collision_payment_refund",
        "description": "High-Collision Interference: Execute multiple billing refunds with similar transaction IDs; ask for refund_id of TXN-88195-REF",
        "interactive_turns": [
            {
                "user": "Process billing refund for transaction TXN-88194-REF.",
                "tool_call": '{"name": "issue_refund", "arguments": {"transaction_id": "TXN-88194-REF", "amount_cents": 14950, "currency": "USD"}}',
                "tool_response": '{"status": "success", "refund_id": "RFD-5501-A", "processed_at": "2026-07-25T14:20:00Z"}'
            },
            {
                "user": "Process billing refund for transaction TXN-88195-REF.",
                "tool_call": '{"name": "issue_refund", "arguments": {"transaction_id": "TXN-88195-REF", "amount_cents": 29900, "currency": "USD"}}',
                "tool_response": '{"status": "success", "refund_id": "RFD-9942-B", "processed_at": "2026-07-25T14:22:00Z"}'
            },
            {
                "user": "Process billing refund for transaction TXN-88200-REF.",
                "tool_call": '{"name": "issue_refund", "arguments": {"transaction_id": "TXN-88200-REF", "amount_cents": 8900, "currency": "USD"}}',
                "tool_response": '{"status": "success", "refund_id": "RFD-1104-C", "processed_at": "2026-07-25T14:25:00Z"}'
            }
        ],
        "query": "What was the exact refund_id returned in the tool response for transaction TXN-88195-REF?",
        "target": "RFD-9942-B"
    },
    {
        "id": "task_3_multi_step_tool_chain",
        "description": "Chained Tool Execution: Output of execute_sql (user_id 98412) is passed to introspect_token; ask for tenant_id of user_id 98412",
        "interactive_turns": [
            {
                "user": "Query customer database for tenant TENANT-GOLD-99.",
                "tool_call": '{"name": "execute_sql", "arguments": {"query": "SELECT user_id FROM user_profiles WHERE tenant_id = \'TENANT-GOLD-99\'", "limit": 1}}',
                "tool_response": '{"status": 200, "rows": [{"user_id": 98412}]}'
            },
            {
                "user": "Now introspect the auth token for the returned user_id 98412.",
                "tool_call": '{"name": "introspect_token", "arguments": {"user_id": 98412, "token_type": "Bearer"}}',
                "tool_response": '{"active": true, "sub": "usr_gold_98412", "scope": "admin:all"}'
            }
        ],
        "query": "Which tenant_id in the earlier SQL call produced the user_id 98412 used in token introspection?",
        "target": "TENANT-GOLD-99"
    },
    {
        "id": "task_4_state_tracking_quota_deduction",
        "description": "State Tracking / Evolving Working Memory: Starting API quota 500000; Turn 1 deducts 50000; Turn 2 deducts 125000; ask for remaining quota",
        "interactive_turns": [
            {
                "user": "Check initial API quota for enterprise tenant TENANT-MEGA.",
                "tool_call": '{"name": "get_api_quota", "arguments": {"tenant_id": "TENANT-MEGA"}}',
                "tool_response": '{"tenant_id": "TENANT-MEGA", "initial_quota": 500000, "remaining_quota": 500000}'
            },
            {
                "user": "Execute batch indexing job consuming 50000 API units.",
                "tool_call": '{"name": "consume_api_quota", "arguments": {"tenant_id": "TENANT-MEGA", "units_consumed": 50000}}',
                "tool_response": '{"tenant_id": "TENANT-MEGA", "remaining_quota": 450000}'
            },
            {
                "user": "Execute large vector embedding sync job consuming 125000 API units.",
                "tool_call": '{"name": "consume_api_quota", "arguments": {"tenant_id": "TENANT-MEGA", "units_consumed": 125000}}',
                "tool_response": '{"tenant_id": "TENANT-MEGA", "remaining_quota": 325000}'
            }
        ],
        "query": "After both batch jobs were executed, what is the exact remaining_quota for TENANT-MEGA?",
        "target": "325000"
    },
    {
        "id": "task_5_state_tracking_replica_update",
        "description": "State Tracking / Deployment Scaling: Initial replicas 4; Turn 1 scale to 16; Turn 2 scale to 24; ask for active target replicas",
        "interactive_turns": [
            {
                "user": "Inspect current Kubernetes deployment scale for cluster us-east-prod-cluster-04.",
                "tool_call": '{"name": "get_deployment", "arguments": {"cluster_id": "us-east-prod-cluster-04", "deployment": "api-gateway"}}',
                "tool_response": '{"cluster_id": "us-east-prod-cluster-04", "current_replicas": 4}'
            },
            {
                "user": "Scale api-gateway deployment to 16 target replicas for morning peak.",
                "tool_call": '{"name": "scale_deployment", "arguments": {"cluster_id": "us-east-prod-cluster-04", "target_replicas": 16}}',
                "tool_response": '{"status": "scaling", "desired_replicas": 16}'
            },
            {
                "user": "Emergency scale api-gateway deployment to 24 target replicas due to traffic spike.",
                "tool_call": '{"name": "scale_deployment", "arguments": {"cluster_id": "us-east-prod-cluster-04", "target_replicas": 24}}',
                "tool_response": '{"status": "scaling", "desired_replicas": 24}'
            }
        ],
        "query": "What is the final target_replicas count specified in the last scale_deployment call for us-east-prod-cluster-04?",
        "target": "24"
    },
    {
        "id": "task_6_tool_output_retrieval",
        "description": "Tool Output Retrieval: Vector search returns document title 'Zero Trust Access Controls' for id 'doc_9912'; ask for exact title",
        "interactive_turns": [
            {
                "user": "Search knowledge base for security policy documents.",
                "tool_call": '{"name": "query_vector_store", "arguments": {"index_name": "kb_architecture_v3", "query_text": "security policy"}}',
                "tool_response": '{"matches": [{"id": "doc_9912", "score": 0.942, "metadata": {"title": "Zero Trust Access Controls", "category": "sec_audit"}}]}'
            }
        ],
        "query": "What was the exact document title returned in the tool response for document id doc_9912?",
        "target": "Zero Trust Access Controls"
    }
]

# ==============================================================================
# 3. PROMPT GENERATOR & STRICT EVALUATION UTILITIES
# ==============================================================================
def clean_and_normalize(val_str: str) -> str:
    """Strips quotes, markdown formatting, spaces, and punctuation for strict exact value comparison."""
    if val_str is None:
        return ""
    s = str(val_str).strip()
    s = re.sub(r'^[\`\'\"*]+|[\`\'\"*]+$', '', s)
    s = re.sub(r'^\*{1,2}|\*{1,2}$', '', s)
    s = s.strip(".,;:()")
    return s.strip()

def check_strict_exact_match(prediction: str, target: str) -> bool:
    """Strict evaluation: extracts normalized pred and target, asserting exact match or integer equality."""
    pred_clean = clean_and_normalize(prediction)
    target_clean = clean_and_normalize(target)
    
    if pred_clean.lower() == target_clean.lower():
        return True
        
    # Check numeric equivalence if applicable (e.g., "24" vs 24 or "325000" vs 325,000)
    try:
        pred_num = int(re.sub(r'[,_]', '', pred_clean))
        target_num = int(re.sub(r'[,_]', '', target_clean))
        if pred_num == target_num:
            return True
    except ValueError:
        pass
        
    return False

def build_interleaved_agent_prompt(tok, eval_task, ctx_len, depth=0.5, is_llama=False):
    """
    Builds a realistic, jagged agent conversation trace:
      - Interleaves target scenario turns at specified context depth (0.1, 0.3, 0.5, 0.7, 0.9).
      - Interleaves diverse agent domain filler turns (no synthetic repetition).
      - Scatter tool calls throughout context.
    """
    target_turns_str = ""
    for turn in eval_task["interactive_turns"]:
        target_turns_str += f"[User]: {turn['user']}\n"
        target_turns_str += f"[Assistant Tool Call]: {turn['tool_call']}\n"
        target_turns_str += f"[Tool Output]: {turn['tool_response']}\n"

    q_str = f"Question: {eval_task['query']} Answer with ONLY the exact parameter or value, with no preamble."

    filler_turns_str = ""
    random_seed_domains = list(DIVERSE_AGENT_DOMAINS)
    random.seed(42)
    random.shuffle(random_seed_domains)

    for domain in random_seed_domains:
        for turn in domain:
            role = "User" if turn['role'] == "user" else ("Assistant Tool Call" if turn['role'] == "assistant" else "Tool Output")
            filler_turns_str += f"[{role}]: {turn['content']}\n"

    filler_toks = tok.encode(filler_turns_str, add_special_tokens=False)
    target_toks = tok.encode(target_turns_str, add_special_tokens=False)
    q_toks = tok.encode(q_str, add_special_tokens=False)

    budget = ctx_len - len(target_toks) - len(q_toks) - 120
    if budget < 100:
        budget = 100

    reps = budget // len(filler_toks) + 1
    all_filler = (filler_toks * reps)[:budget]

    split_at = int(len(all_filler) * depth)
    p1 = tok.decode(all_filler[:split_at])
    p2 = tok.decode(all_filler[split_at:])

    user_content = p1 + target_turns_str + p2 + "\n\n" + q_str

    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(
                [
                    {"role": "system", "content": "You are an enterprise AI agent. Maintain working memory across multi-turn tool execution logs."},
                    {"role": "user", "content": user_content}
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    if is_llama:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an enterprise AI agent.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + user_content + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        "<|im_start|>system\nYou are an enterprise AI agent.<|im_end|>\n"
        "<|im_start|>user\n" + user_content + "<|im_end|>\n<|im_start|>assistant\n"
    )

# ==============================================================================
# 4. BENCHMARK RUNNER
# ==============================================================================
def run_agent_memory_benchmark(model_id, ctx_list, depths_list=[0.1, 0.3, 0.5, 0.7, 0.9], compressed=True):
    from serving.mlx_dkv_wrapper import MLXDKVWrapper
    print(f"[Agent Memory Benchmark v2.0] Loading {model_id} (Compressed={compressed})...")

    env_backup = os.environ.get("DKV_COMPRESSED_DECODE")
    if not compressed:
        os.environ["DKV_COMPRESSED_DECODE"] = "0"
    else:
        os.environ["DKV_COMPRESSED_DECODE"] = "1"

    wrapper = MLXDKVWrapper(model_id, config={"rank": 32})
    tok = wrapper.tokenizer
    is_llama = "llama" in model_id.lower()

    summary_results = {}

    for ctx in ctx_list:
        summary_results[ctx] = {"depths": {}}
        ctx_recalled_total = 0
        ctx_tasks_total = 0

        print(f"\n========================================================")
        print(f" Context Length: {ctx} Tokens (Engine Mode: {'DKV Compressed' if compressed else 'Dense Baseline'})")
        print(f" Depth Sweep: {depths_list}")
        print(f"========================================================")

        for depth in depths_list:
            recalled = 0
            total = len(AGENT_EVAL_TASKS)
            task_results = []

            for i, task in enumerate(AGENT_EVAL_TASKS):
                sid = f"agent_v2_{ctx}_d{int(depth*100)}_{i}"
                prompt = build_interleaved_agent_prompt(tok, task, ctx, depth=depth, is_llama=is_llama)

                try:
                    wrapper.clear_session(sid)
                except Exception:
                    pass

                t0 = time.time()
                out = wrapper.generate(prompt, max_new_tokens=32, temperature=0.0, session_id=sid)
                elapsed = time.time() - t0

                # Extract strictly the new generated tokens after the prompt question
                if task["query"] in out:
                    gen_text = out.split(task["query"])[-1].strip()
                elif "assistant\n" in out:
                    gen_text = out.split("assistant\n")[-1].strip()
                else:
                    gen_text = out.strip()

                if "assistant\n" in gen_text:
                    gen_text = gen_text.split("assistant\n")[-1].strip()
                if "no preamble." in gen_text:
                    gen_text = gen_text.split("no preamble.")[-1].strip()

                # Clean up any trailing control tags or template artifacts
                gen_text = gen_text.split("<|im_end|>")[0].split("<|eot_id|>")[0].strip()

                pass_strict = check_strict_exact_match(gen_text, task["target"])
                if pass_strict:
                    recalled += 1
                    status = "PASS [STRICT]"
                else:
                    status = "FAIL"

                print(f"[{i+1}/{total}] {task['id']} -> {status} ({elapsed:.1f}s)")
                print(f"    Target: {task['target']!r}")
                print(f"    Prediction: {gen_text!r}")

                task_results.append({
                    "task_id": task["id"],
                    "status": status,
                    "target": task["target"],
                    "prediction": gen_text,
                    "latency_sec": round(elapsed, 2)
                })

            acc = (recalled / total) * 100.0
            depth_key = f"{int(depth*100)}%"
            summary_results[ctx]["depths"][depth_key] = {
                "recalled": recalled,
                "total": total,
                "accuracy": acc,
                "tasks": task_results
            }
            ctx_recalled_total += recalled
            ctx_tasks_total += total
            print(f"  Depth {depth_key}: {recalled}/{total} Passed ({acc:.1f}%)")

        overall_ctx_acc = (ctx_recalled_total / ctx_tasks_total) * 100.0 if ctx_tasks_total > 0 else 0.0
        summary_results[ctx]["mean_accuracy"] = overall_ctx_acc
        print(f"--> Ctx {ctx} Overall Mean Accuracy: {overall_ctx_acc:.1f}%")

    if env_backup is not None:
        os.environ["DKV_COMPRESSED_DECODE"] = env_backup

    return summary_results

def main():
    parser = argparse.ArgumentParser(description="Agent Memory Benchmark v2.0 — Multi-Turn Tool-Calling & State Tracking")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen2.5-1.5B-Instruct-4bit", help="Model ID or local path (any MLX or HF model)")
    parser.add_argument("--ctx", nargs="+", type=int, default=[8192, 16384, 32768, 65536], help="Context lengths to evaluate")
    parser.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9], help="Target state depth fractions to sweep")
    parser.add_argument("--mode", choices=["compressed", "dense", "both"], default="compressed", help="Evaluation mode")
    parser.add_argument("--output", type=str, default="benchmarks/results/agent_eval_latest.json", help="Path to save machine-readable results JSON")
    args = parser.parse_args()

    results = {}
    if args.mode in ("compressed", "both"):
        print("\n=== Running DKV Compressed Agent Memory Mode ===")
        results["compressed"] = run_agent_memory_benchmark(args.model, args.ctx, depths_list=args.depths, compressed=True)

    if args.mode in ("dense", "both"):
        print("\n=== Running Dense Baseline Agent Memory Mode ===")
        results["dense"] = run_agent_memory_benchmark(args.model, args.ctx, depths_list=args.depths, compressed=False)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n================== FINAL BENCHMARK SUMMARY ==================")
    print(f"Results saved to: {args.output}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
