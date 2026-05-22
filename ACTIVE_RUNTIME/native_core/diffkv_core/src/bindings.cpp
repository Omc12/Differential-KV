// diffkv_core/src/bindings.cpp
// PyBind11 bindings for libdiffkv_core.
// Exposes DiffKVBlockStateTable, DiffKVCompressorThread, DiffKVPagingStream
// to Python with minimal surface area. Python never owns synchronization.

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include "block_state.hpp"
#include "compressor_thread.hpp"
#include "paging_stream.hpp"

namespace py = pybind11;
using namespace diffkv;

PYBIND11_MODULE(diffkv_core, m) {
    m.doc() = "Differential KV native runtime core — C++/CUDA extension";

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

    // ── SlabTier enum ────────────────────────────────────────────────────────
    py::enum_<SlabTier>(m, "SlabTier")
        .value("Rank8",  SlabTier::Rank8)
        .value("Rank16", SlabTier::Rank16)
        .value("Rank32", SlabTier::Rank32)
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
}
