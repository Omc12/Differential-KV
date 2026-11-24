// Variant D: leave all three raise-only gates alone, add ONE final clamp
//   srl_k_keep = min(srl_k_keep, n_comp_blocks + growth)
// after them. Attending more blocks than exist is meaningless, and sizing a
// cache for them is pure waste -- but the clamp must leave room for blocks that
// get compressed DURING generation as the dense window flushes, or the cache
// under-allocates mid-answer. growth = ceil(max_new / mbs).
#include <algorithm>
#include <cstdio>
#include <string>

int pipeline(int n_ctx, int mbs, int L, int recency, const char* preset,
             bool final_clamp, int max_new, int* nc_out) {
    int k = 16;
    k = std::max(k, std::max(16, 1024 / mbs));
    int n_slots = n_ctx / mbs;
    if (preset) {
        if (!std::string(preset).compare("low"))  n_slots = 4096  / mbs;
        if (!std::string(preset).compare("mid"))  n_slots = 8192  / mbs;
        if (!std::string(preset).compare("high")) n_slots = 16384 / mbs;
    }
    k = std::min(k, n_slots);
    int nc = std::max(0, (L - recency)) / mbs;
    if (nc > n_slots) nc = n_slots;
    *nc_out = nc;
    if (nc > 0) {
        int amin = std::max(20, (int)(0.15f * nc));
        int ak = std::min(std::max(amin, std::min(k, std::min(200, nc))), 256);
        if (ak > k) k = ak;
        if (n_slots <= 32) { int t = std::min(n_slots, 256); if (t > k) k = t; }
        if (nc <= 36)      { int t = std::min(std::min(nc, n_slots), 256); if (t > k) k = t; }
        if (final_clamp) {
            int growth = (max_new + mbs - 1) / mbs;
            k = std::min(k, std::min(nc + growth, n_slots));
        }
    }
    return k;
}

int main() {
    const int mbs = 1024, n_ctx = 32768, rec = 512, max_new = 256;
    const int Ls[] = { 4096, 8192, 16384, 32768 };
    const char* ps[] = { nullptr, "mid", "high" };
    const char* pn[] = { "(none)", "mid", "high" };
    printf("mbs=%d n_ctx=%d recency=%d max_new=%d  (growth allowance = %d block)\n\n",
           mbs, n_ctx, rec, max_new, (max_new + mbs - 1) / mbs);
    printf("%-7s %7s %6s | %4s %8s | %4s %8s | %8s %7s\n",
           "preset", "L", "n_comp", "K", "cap", "K_D", "cap_D", "need", "saved");
    printf("%s\n", std::string(74, '-').c_str());
    long tot_before = 0, tot_after = 0;
    for (int pi = 0; pi < 3; ++pi) for (int L : Ls) {
        int nc1, nc2;
        int a = pipeline(n_ctx, mbs, L, rec, ps[pi], false, max_new, &nc1);
        int d = pipeline(n_ctx, mbs, L, rec, ps[pi], true,  max_new, &nc2);
        int capa = a * (mbs + 1), capd = d * (mbs + 1), need = nc1 * (mbs + 1);
        tot_before += capa; tot_after += capd;
        printf("%-7s %7d %6d | %4d %8d | %4d %8d | %8d %6.1f%%\n",
               pn[pi], L, nc1, a, capa, d, capd, need,
               capa ? 100.0 * (capa - capd) / capa : 0.0);
    }
    printf("\ntotal rows/layer across the grid: %ld -> %ld  (%.1f%% less)\n",
           tot_before, tot_after, 100.0 * (tot_before - tot_after) / tot_before);
    printf("fp16 KV at head_dim=128, kv_heads=8, 28 layers, K+V:\n");
    double bpr = 128.0 * 8 * 2 * 2 * 28 / (1024 * 1024);
    printf("  %.0f MB -> %.0f MB\n", tot_before / 12.0 * bpr, tot_after / 12.0 * bpr);
    return 0;
}
