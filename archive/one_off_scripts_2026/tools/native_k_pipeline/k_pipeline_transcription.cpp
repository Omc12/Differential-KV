// Standalone transcription of the native srl_k_keep -> decode-cache pipeline.
// Faithful to dkv_native/src/main.cpp at commit 1157f3a2:
//   2117  srl_k_keep = 16
//   2153  N4.2 floor        max(16, 1024 / mbs)
//   2239  n_slots           n_ctx / mbs   (preset overrides at 2263-2270)
//   2281  clamp             min(srl_k_keep, n_slots)
//   4446  adaptive-k        max(max(20,0.15*nc), min(k, min(200,nc))), cap 256, RAISE only
//   4469  n_slots<=32       min(n_slots,256),                          RAISE only
//   4484  n_comp<=36        min(min(nc,n_slots),256),                   RAISE only
//   4607  cache_routed_cap  srl_k_keep * (mbs + 1)
// No ggml, no model: this settles the ALLOCATION ARITHMETIC only.
#include <algorithm>
#include <cstdio>
#include <string>

struct R { int k; int cap; int need; };

static R pipeline(int n_ctx, int mbs, int L, int recency,
                  const char* preset, bool fix_nslots, bool fix_adaptive) {
    int k = 16;                                              // 2117
    k = std::max(k, std::max(16, 1024 / mbs));               // 2153 N4.2 floor

    int n_slots = n_ctx / mbs;                               // 2239
    if (preset) {                                            // 2263-2270
        if (!std::string(preset).compare("low"))  n_slots = 4096  / mbs;
        if (!std::string(preset).compare("mid"))  n_slots = 8192  / mbs;
        if (!std::string(preset).compare("high")) n_slots = 16384 / mbs;
    }
    k = std::min(k, n_slots);                                // 2281

    // Compressed blocks that actually exist for a prompt of L tokens: everything
    // outside the recency window, in units of mbs.
    int nc = std::max(0, (L - recency)) / mbs;
    if (nc > n_slots) nc = n_slots;                          // pool cannot exceed slots

    if (nc > 0) {
        // 4446 adaptive-k (fast pruning, the DEFAULT branch)
        int amin = fix_adaptive ? std::min(20, nc) : std::max(20, (int)(0.15f * nc));
        int amax = std::min(200, nc);
        int ak   = std::max(amin, std::min(k, amax));
        ak = std::min(ak, 256);
        if (ak > k) k = ak;                                  // RAISE only

        // 4469 the fallback under review
        int gate = fix_nslots ? nc : n_slots;
        if (gate <= 32) {
            int t = std::min(fix_nslots ? nc : n_slots, 256);
            if (t > k) k = t;                                // RAISE only
        }
        // 4484 already keys off n_comp, but also RAISE only
        if (nc <= 36 && nc > 0) {
            int t = std::min(std::min(nc, n_slots), 256);
            if (t > k) k = t;                                // RAISE only
        }
    }
    return { k, k * (mbs + 1), nc * (mbs + 1) };
}

int main() {
    const int mbs = 1024, n_ctx = 32768, rec = 512;
    const int Ls[] = { 4096, 8192, 16384, 32768 };
    const char* presets[] = { nullptr, "mid", "high" };
    const char* pnames[]  = { "(none)", "mid", "high" };

    printf("micro_block_size=%d  n_ctx=%d  recency=%d\n", mbs, n_ctx, rec);
    printf("cap = srl_k_keep*(mbs+1) rows/layer;  need = n_comp*(mbs+1)\n\n");
    printf("%-7s %7s %6s | %4s %8s %8s %6s | %4s %8s %6s | %4s %8s %6s\n",
           "preset", "L", "n_comp", "K", "cap", "need", "waste",
           "K'", "cap'", "waste'", "K''", "cap''", "waste''");
    printf("%s\n", std::string(112, '-').c_str());
    for (int pi = 0; pi < 3; ++pi) {
        for (int L : Ls) {
            R a = pipeline(n_ctx, mbs, L, rec, presets[pi], false, false); // shipped
            R b = pipeline(n_ctx, mbs, L, rec, presets[pi], true,  false); // fix n_slots only
            R c = pipeline(n_ctx, mbs, L, rec, presets[pi], true,  true);  // + adaptive floor
            auto w = [](R r) { return r.need ? (double)r.cap / r.need : 0.0; };
            printf("%-7s %7d %6d | %4d %8d %8d %5.1fx | %4d %8d %5.1fx | %4d %8d %5.1fx\n",
                   pnames[pi], L, a.need / (mbs + 1),
                   a.k, a.cap, a.need, w(a), b.k, b.cap, w(b), c.k, c.cap, w(c));
        }
    }
    printf("\nK   = shipped\nK'  = 4469 gate keyed off n_comp_blocks\n"
           "K'' = K' plus adaptive_k_min floor min(20,n_comp) instead of max(20,..)\n");
    return 0;
}
