import json

d = json.load(open("benchmark_results_ollama.json"))
contexts = [1024, 2048, 4096, 8192, 16384]
modes = ["dense_pytorch", "diffkv_mlx", "ollama_fp16", "ollama_quant"]

for size in ["0.5b", "1.5b"]:
    print(f"\n=== SIZE: {size} ===")
    results_data = d[size]
    
    prefill = {m: [] for m in modes}
    tps = {m: [] for m in modes}
    acc = {m: [] for m in modes}
    mem = {m: [] for m in modes}
    
    for c in contexts:
        for m in modes:
            m_res = results_data.get(m, {}).get(str(c), {})
            if "error" in m_res or not m_res:
                prefill[m].append(None)
                tps[m].append(None)
                acc[m].append(None)
                mem[m].append(None)
            else:
                prefill[m].append(m_res.get("prefill_s"))
                tps[m].append(m_res.get("decode_tps"))
                acc[m].append(m_res.get("accuracy"))
                peak_rss = m_res.get("peak_rss_mb", 0.0)
                peak_res = m_res.get("peak_reserved_mb", 0.0)
                mem[m].append(max(peak_rss, peak_res) / 1024.0)
                
    for m in modes:
        print(f"Mode: {m}")
        print(f"  Prefill:  {prefill[m]}")
        print(f"  TPS:      {tps[m]}")
        print(f"  Accuracy: {acc[m]}")
        print(f"  Memory:   {mem[m]}")
