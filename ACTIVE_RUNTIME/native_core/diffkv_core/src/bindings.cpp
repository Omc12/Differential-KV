// diffkv_core/src/bindings.cpp
// PyBind11 bindings for libdiffkv_core.
//
// Exposes:
//   Phase 0 (existing): DiffKVBlockStateTable
//                       DiffKVCompressorThread (CUDA only)
//                       DiffKVPagingStream     (CUDA only)
//   Phase 1 (new):      compute_query_desc, semantic_search_topk, anchor_screen,
//                       decode_attention_aten, decode_attention_aten_lse
//   Phase 1 Mac:        DiffKVCompressorThreadCPU (Accelerate LAPACK, no Metal)
//
// Python never owns synchronization — all atomic/mutex/thread state lives in C++.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "block_state.hpp"
#include "srl_router.hpp"
#include "decode_attention.hpp"

#ifdef DIFFKV_CUDA
#include "compressor_thread.hpp"
#include "paging_stream.hpp"
#endif

#ifdef DIFFKV_APPLE
#include "compressor_cpu.hpp"
#include "metal_runtime.hpp"
#endif

namespace py = pybind11;
using namespace diffkv;

PYBIND11_MODULE(diffkv_core, m) {
    m.doc() = "Differential KV native runtime core — C++/CUDA/Metal extension";

    // ── BlockState enum ──────────────────────────────────────────────────────
    py::enum_<BlockState>(m, "BlockState")
        .value("DenseResident",      BlockState::DenseResident)
        .value("Compressing",        BlockState::Compressing)
        .value("CompressedResident", BlockState::CompressedResident)
        .value("PagingOut",          BlockState::PagingOut)
        .value("CPUResident",        BlockState::CPUResident)
        .value("Reloading",          BlockState::Reloading)
        .value("Invalid",            BlockState::Invalid)
        .value("Freed",              BlockState::Freed)
        .export_values();

    // ── DiffKVBlockStateTable ────────────────────────────────────────────────
    py::class_<DiffKVBlockStateTable>(m, "DiffKVBlockStateTable")
        .def(py::init<>())
        .def("transition", &DiffKVBlockStateTable::transition,
             py::arg("block_id"), py::arg("expected"), py::arg("desired"),
             "CAS-based atomic state transition. Raises on illegal transition.")
        .def("force_invalidate", &DiffKVBlockStateTable::force_invalidate,
             py::arg("block_id"),
             "Force block to Invalid state — safe to call from any thread on disconnect.")
        .def("get", &DiffKVBlockStateTable::get,
             py::arg("block_id"),
             "Lock-free read of current block state.")
        .def("are_replay_safe", [](const DiffKVBlockStateTable& self,
                                    const std::vector<uint32_t>& block_ids) {
            return self.are_replay_safe(block_ids.data(), block_ids.size());
        }, py::arg("block_ids"),
           "Returns True iff ALL blocks are CompressedResident or DenseResident — graph replay guard.");

    // ── CUDA-only classes (compressor + paging) ──────────────────────────────
#ifdef DIFFKV_CUDA
    // ── SlabTier enum ────────────────────────────────────────────────────────
    py::enum_<SlabTier>(m, "SlabTier")
        .value("Rank8",  SlabTier::Rank8)
        .value("Rank16", SlabTier::Rank16)
        .value("Rank32", SlabTier::Rank32)
        .export_values();

    // ── CompressJob ──────────────────────────────────────────────────────────
    py::class_<CompressJob>(m, "CompressJob")
        .def(py::init<>())
        .def_readwrite("block_id",     &CompressJob::block_id)
        .def_readwrite("session_id",   &CompressJob::session_id)
        .def_readwrite("block_size",   &CompressJob::block_size)
        .def_readwrite("heads",        &CompressJob::heads)
        .def_readwrite("head_dim",     &CompressJob::head_dim)
        .def_readwrite("target_slab",  &CompressJob::target_slab);
        // Note: GPU raw pointers are set from C++ bridge layer, not Python.

    // ── DiffKVCompressorThread ───────────────────────────────────────────────
    py::class_<DiffKVCompressorThread>(m, "DiffKVCompressorThread")
        .def(py::init<DiffKVBlockStateTable&, SessionAliveCallback>(),
             py::arg("state_table"), py::arg("alive_callback"),
             "alive_callback(session_id: int) -> bool")
        .def("start", &DiffKVCompressorThread::start,
             "Start the native worker thread.")
        .def("stop", &DiffKVCompressorThread::stop,
             "Stop and join the native worker thread.")
        .def("submit", &DiffKVCompressorThread::submit,
             py::arg("job"),
             "Non-blocking job submission. Returns False on queue overflow — block stays DenseResident.")
        .def_property_readonly("jobs_processed",  &DiffKVCompressorThread::jobs_processed)
        .def_property_readonly("jobs_dropped",    &DiffKVCompressorThread::jobs_dropped)
        .def_property_readonly("queue_overflows", &DiffKVCompressorThread::queue_overflows)
        .def_property_readonly("queue_depth",     &DiffKVCompressorThread::queue_depth);

    // ── PagingJob ────────────────────────────────────────────────────────────
    py::class_<PagingJob>(m, "PagingJob")
        .def(py::init<>())
        .def_readwrite("block_id",   &PagingJob::block_id)
        .def_readwrite("session_id", &PagingJob::session_id)
        .def_readwrite("byte_size",  &PagingJob::byte_size)
        .def_readwrite("is_reload",  &PagingJob::is_reload);
        // GPU/CPU raw ptrs set from C++ bridge, not Python.

    // ── DiffKVPagingStream ───────────────────────────────────────────────────
    py::class_<DiffKVPagingStream>(m, "DiffKVPagingStream")
        .def(py::init<DiffKVBlockStateTable&, int>(),
             py::arg("state_table"), py::arg("cuda_device") = 0)
        .def("issue_eviction", &DiffKVPagingStream::issue_eviction,
             py::arg("job"),
             "Issue async D2H transfer. Returns False if block not CompressedResident.")
        .def("issue_reload", &DiffKVPagingStream::issue_reload,
             py::arg("job"),
             "Issue async H2D transfer. Returns False if block not CPUResident.")
        .def("poll_completions", &DiffKVPagingStream::poll_completions,
             "Poll pending transfers; advance state machine for completed ones. Call between batch steps.")
        .def_property_readonly("evictions_completed",  &DiffKVPagingStream::evictions_completed)
        .def_property_readonly("reloads_completed",    &DiffKVPagingStream::reloads_completed)
        .def_property_readonly("cancelled_transfers",  &DiffKVPagingStream::cancelled_transfers);
#endif  // DIFFKV_CUDA

    // ── Apple/Mac-only CPU compressor (Accelerate LAPACK, zero Metal) ─────────
#ifdef DIFFKV_APPLE
    py::class_<CompressJobCPU>(m, "CompressJobCPU")
        .def(py::init<>())
        .def_readwrite("block_id",    &CompressJobCPU::block_id)
        .def_readwrite("session_id",  &CompressJobCPU::session_id)
        .def_readwrite("block_size",  &CompressJobCPU::block_size)
        .def_readwrite("feat_dim",    &CompressJobCPU::feat_dim)
        .def_readwrite("rank",        &CompressJobCPU::rank);
        // Raw pointers (dense_k_ptr, out_u_ptr, etc.) are set from C++ bridge, not Python.

    py::class_<DiffKVCompressorThreadCPU>(m, "DiffKVCompressorThreadCPU")
        .def(py::init<DiffKVBlockStateTable&, CPUSessionAliveCallback>(),
             py::arg("state_table"), py::arg("alive_callback"),
             "alive_callback(session_id: int) -> bool\n"
             "Accelerate LAPACK SVD — no Metal/MPS calls, AMX-accelerated on Apple Silicon.")
        .def("start", &DiffKVCompressorThreadCPU::start,
             "Start the CPU compressor worker thread.")
        .def("stop", &DiffKVCompressorThreadCPU::stop,
             "Stop and join the worker thread.")
        .def("submit", &DiffKVCompressorThreadCPU::submit,
             py::arg("job"),
             "Non-blocking submission. Returns False on queue overflow.")
        .def_property_readonly("jobs_processed",  &DiffKVCompressorThreadCPU::jobs_processed)
        .def_property_readonly("jobs_dropped",    &DiffKVCompressorThreadCPU::jobs_dropped)
        .def_property_readonly("queue_overflows", &DiffKVCompressorThreadCPU::queue_overflows)
        .def_property_readonly("queue_depth",     &DiffKVCompressorThreadCPU::queue_depth);
#endif  // DIFFKV_APPLE

    // ── Phase 1: SRL Routing functions ───────────────────────────────────────
    // Stateless free functions — no class, no shared state.

    m.def("compute_query_desc",
        &diffkv::compute_query_desc,
        py::arg("Q"),
        py::arg("W_proj"),
        R"doc(
Compute a compact [desc_dim] float32 descriptor for a query tensor.

Replaces: native_core/srl/chunk_descriptor.py::compute_query_descriptor()

Args:
    Q      : torch.Tensor [H, D]        all query heads (float16 or float32)
    W_proj : torch.Tensor [desc_dim, D] random projection matrix (float32)

Returns:
    torch.Tensor [desc_dim] float32, L2-normalized.
)doc"
    );

    m.def("semantic_search_topk",
        &diffkv::semantic_search_topk,
        py::arg("q_desc"),
        py::arg("desc_matrix"),
        py::arg("k"),
        R"doc(
Dot-product ANN search over descriptor matrix — returns top-k row indices.

Replaces: native_core/srl/semantic_index.py::SemanticIndex.search()

Args:
    q_desc      : torch.Tensor [desc_dim]    normalized query descriptor
    desc_matrix : torch.Tensor [N, desc_dim] pool descriptor matrix
    k           : int                         number of top slots to return

Returns:
    torch.Tensor [min(k,N)] int64 — row indices into desc_matrix.
)doc"
    );

    m.def("anchor_screen",
        &diffkv::anchor_screen,
        py::arg("Q"),
        py::arg("anchors_K"),
        py::arg("candidate_slots"),
        py::arg("scale"),
        py::arg("k_keep"),
        R"doc(
Level-1 anchor screening: rerank candidate pool slots by anchor key dot product.

Replaces: native_core/srl/query_router.py::two_level_gate()

Args:
    Q               : torch.Tensor [H, D]            all query heads
    anchors_K       : torch.Tensor [N_pool, kv_heads, D]  pool.anchors_K
    candidate_slots : torch.Tensor [M] int32          candidate slot IDs
    scale           : float                            attention scale
    k_keep          : int                              slots to keep

Returns:
    torch.Tensor [min(k_keep, M)] int32 — top slot IDs by anchor score.
)doc"
    );

    // ── Phase 1: Decode Attention functions ───────────────────────────────────

    m.def("decode_attention_aten",
        &diffkv::decode_attention_aten,
        py::arg("Q"),
        py::arg("U_pool"),
        py::arg("U_scale_pool"),
        py::arg("VK_pool"),
        py::arg("VV_pool"),
        py::arg("anchors_K"),
        py::arg("anchors_V"),
        py::arg("seq_lens"),
        py::arg("scales"),
        py::arg("cos_anc"),
        py::arg("sin_anc"),
        py::arg("slot_indices"),
        py::arg("scale"),
        py::arg("n_q_heads"),
        py::arg("n_kv_heads"),
        py::arg("rank"),
        R"doc(
Fused Project-Then-Attend decode attention (ATen C++ API, no Python GIL on hot path).

Replaces: fused_decode_mps() in native_core/sparse_decode/triton_fused_decode.py

Args:
    Q            : [H_q, D]                  float16
    U_pool       : [N_pool, S_max, R]         int8    pool.U
    U_scale_pool : [N_pool]                   float16 pool.U_scale
    VK_pool      : [N_pool, R, kv_heads, D]   float16 pool.V_K
    VV_pool      : [N_pool, R, kv_heads, D]   float16 pool.V_V
    anchors_K    : [N_pool, kv_heads, D]      float16 pool.anchors_K
    anchors_V    : [N_pool, kv_heads, D]      float16 pool.anchors_V
    seq_lens     : [N_pool]                   int32   pool.seq_lens
    scales       : [N_pool]                   float16 pool.scales
    cos_anc      : [K_active, D]              float32 RoPE cosine at anchor positions
    sin_anc      : [K_active, D]              float32 RoPE sine at anchor positions
    slot_indices : [K_active]                 int32   selected slots from SRL
    scale        : float  (1 / sqrt(head_dim))
    n_q_heads    : int
    n_kv_heads   : int
    rank         : int   (SVD rank R)

Returns:
    torch.Tensor [H_q, D] float16 — attention output.
)doc"
    );

    m.def("decode_attention_aten_lse",
        &diffkv::decode_attention_aten_lse,
        py::arg("Q"),
        py::arg("U_pool"),
        py::arg("U_scale_pool"),
        py::arg("VK_pool"),
        py::arg("VV_pool"),
        py::arg("anchors_K"),
        py::arg("anchors_V"),
        py::arg("seq_lens"),
        py::arg("scales"),
        py::arg("cos_anc"),
        py::arg("sin_anc"),
        py::arg("slot_indices"),
        py::arg("scale"),
        py::arg("n_q_heads"),
        py::arg("n_kv_heads"),
        py::arg("rank"),
        R"doc(
Fused Project-Then-Attend decode attention — returns output AND log-sum-exp.

Use this variant when combining sparse-history output with dense-window SDPA
via LSE combine (i.e., when both dense_blocks and compressed blocks are present).

Returns:
    Tuple of:
      torch.Tensor [H_q, D] float16 — attention output
      torch.Tensor [H_q]    float32 — log-sum-exp per query head
)doc"
    );

    m.def("fused_decode_attention_combined",
        &diffkv::fused_decode_attention_combined,
        py::arg("Q"),
        py::arg("dense_k"),
        py::arg("dense_v"),
        py::arg("cos_dense"),
        py::arg("sin_dense"),
        py::arg("U_pool"),
        py::arg("U_scale_pool"),
        py::arg("VK_pool"),
        py::arg("VV_pool"),
        py::arg("anchors_K"),
        py::arg("anchors_V"),
        py::arg("seq_lens"),
        py::arg("scales"),
        py::arg("cos_anc"),
        py::arg("sin_anc"),
        py::arg("slot_indices"),
        py::arg("scale"),
        py::arg("n_q_heads"),
        py::arg("n_kv_heads"),
        py::arg("rank"),
        R"doc(
Fuses RoPE slicing, dense attention, dense LSE, sparse Metal shader attention, and LSE combination into a single C++ call.
)doc"
    );

    // ── Version / platform capability flags ───────────────────────────────────
    m.attr("__version__")        = "1.1.0";
    m.attr("HAS_DECODE_ATTN")    = true;
    m.attr("HAS_SRL_ROUTER")     = true;
#ifdef DIFFKV_CUDA
    m.attr("HAS_CUDA_PAGING")    = true;
    m.attr("HAS_CPU_COMPRESSOR") = false;
    m.attr("HAS_METAL_ATTN")     = false;
#else
    m.attr("HAS_CUDA_PAGING")    = false;
#ifdef DIFFKV_APPLE
    m.attr("HAS_CPU_COMPRESSOR") = true;
    m.attr("HAS_METAL_ATTN")     = true;

    m.def("decode_attention_metal",
        &diffkv::decode_attention_metal,
        py::arg("Q"),
        py::arg("U_pool"),
        py::arg("U_scale_pool"),
        py::arg("VK_pool"),
        py::arg("VV_pool"),
        py::arg("anchors_K"),
        py::arg("anchors_V"),
        py::arg("seq_lens"),
        py::arg("scales"),
        py::arg("cos_anc"),
        py::arg("sin_anc"),
        py::arg("slot_indices"),
        py::arg("scale"),
        py::arg("n_q_heads"),
        py::arg("n_kv_heads"),
        py::arg("rank"),
        R"doc(
Launch custom Metal Compute Shader for fused Project-Then-Attend decode attention.
)doc"
    );
#else
    m.attr("HAS_CPU_COMPRESSOR") = false;
    m.attr("HAS_METAL_ATTN")     = false;
#endif
#endif

}
