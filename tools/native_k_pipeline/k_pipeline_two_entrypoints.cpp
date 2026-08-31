// Do main.cpp and batch_engine.cpp derive the SAME srl_k_keep?
//
// They are two entry points into one engine, each carrying its own copy of the
// gate chain, and they have diverged before -- micro_block_size 256 vs 1024, which
// meant every batch-served session ran at the block size linkbench scores 9/24 on
// while single-stream ran 1024. Nothing at runtime reconciles them.
//
// A source-level test can assert both files CONTAIN a clamp. Only executing both
// pipelines can show they AGREE. This transcribes each and sweeps the grid.
//
// Build: g++ -O2 -std=c++17 -o k2 k_pipeline_two_entrypoints.cpp && ./k2
#include <algorithm>
#include <cstdio>
#include <string>

static int n_slots_for(int n_ctx, int mbs, const char* preset) {
    int n = n_ctx / mbs;                                  // main.cpp:2245 / batch:703
    if (preset) {
        std::string p(preset);
        if (p == "low")  n = 4096  / mbs;
        if (p == "mid")  n = 8192  / mbs;
        if (p == "high") n = 16384 / mbs;
    }
    return n;
}

// dkv_native/src/main.cpp -- gates at ~4446/4475/4490, clamp after 4504.
static int k_main(int n_ctx, int mbs, int L, int recency, const char* preset,
                  int max_generate, int* nc_out) {
    int n_slots = n_slots_for(n_ctx, mbs, preset);
    int k = std::min(16, n_slots);                        // main.cpp:2117 + 2287
    int nc = std::max(0, L - recency) / mbs;
    if (nc > n_slots) nc = n_slots;
    *nc_out = nc;
    if (nc > 0) {
        int amin = std::max(20, (int)(0.15f * nc));
        int amax = std::min(200, nc);
        int ak = std::min(std::max(amin, std::min(k, amax)), 256);
        if (ak > k) k = ak;
        if (n_slots <= 32) { int t = std::min(n_slots, 256); if (t > k) k = t; }
        if (nc <= 36)      { int t = std::min(std::min(nc, n_slots), 256); if (t > k) k = t; }
        int growth = (max_generate + mbs - 1) / mbs;      // the shipped clamp
        k = std::min(k, std::min(nc + growth, n_slots));
    }
    return k;
}

// dkv_native/serving/batch_engine.cpp -- gates at ~1355/1380/1391, clamp after.
static int k_batch(int n_ctx, int mbs, int L, int recency, const char* preset,
                   int max_tokens, int* nc_out) {
    int n_slots = n_slots_for(n_ctx, mbs, preset);
    int k = std::min(16, n_slots);                        // batch:737 + 746
    int nc = std::max(0, L - recency) / mbs;
    if (nc > n_slots) nc = n_slots;
    *nc_out = nc;
    if (nc > 0) {
        int amin = std::max(20, (int)(0.15f * nc));
        int amax = std::min(200, nc);
        int ak = std::min(std::max(amin, std::min(k, amax)), 256);
        if (ak > k) k = ak;
        if (n_slots <= 32) { int t = std::min(n_slots, 256); if (t > k) k = t; }
        if (nc <= 36)      { int t = std::min(std::min(nc, n_slots), 256); if (t > k) k = t; }
        int growth = (max_tokens + mbs - 1) / mbs;
        k = std::min(k, std::min(nc + growth, n_slots));
    }
    return k;
}

int main() {
    const int Ls[]   = { 1024, 4096, 8192, 16384, 32768, 65536 };
    const int MBS[]  = { 256, 512, 1024, 2048 };
    const int GENS[] = { 40, 256, 2048, 8192 };
    const char* ps[] = { nullptr, "low", "mid", "high" };
    const char* pn[] = { "(none)", "low", "mid", "high" };
    const int n_ctx = 32768, rec = 512;

    long cases = 0, mismatches = 0;
    for (int pi = 0; pi < 4; ++pi)
      for (int mbs : MBS)
        for (int L : Ls)
          for (int g : GENS) {
            int nca, ncb;
            int a = k_main (n_ctx, mbs, L, rec, ps[pi], g, &nca);
            int b = k_batch(n_ctx, mbs, L, rec, ps[pi], g, &ncb);
            ++cases;
            if (a != b || nca != ncb) {
                ++mismatches;
                printf("MISMATCH preset=%-6s mbs=%-5d L=%-6d gen=%-5d | "
                       "main K=%d (nc=%d)  batch K=%d (nc=%d)\n",
                       pn[pi], mbs, L, g, a, nca, b, ncb);
            }
            // Neither entry point may ever keep more blocks than the pool holds,
            // nor more than can exist. This is the property the clamp exists for.
            int n_slots = n_slots_for(n_ctx, mbs, ps[pi]);
            int ceiling = std::min(nca + (g + mbs - 1) / mbs, n_slots);
            if (nca > 0 && a > ceiling) {
                ++mismatches;
                printf("OVER-ALLOC preset=%-6s mbs=%-5d L=%-6d gen=%-5d | "
                       "main K=%d > ceiling %d\n", pn[pi], mbs, L, g, a, ceiling);
            }
          }
    printf("\n%ld cases, %ld mismatches -> %s\n", cases, mismatches,
           mismatches ? "DIVERGENT" : "the two entry points agree everywhere");
    return mismatches ? 1 : 0;
}
