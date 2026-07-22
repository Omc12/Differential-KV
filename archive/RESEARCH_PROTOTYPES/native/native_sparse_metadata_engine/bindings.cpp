// bindings.cpp — pybind11 for NativeSparseMetadataEngine
// RCO-N Phase 41.1

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "native_sparse_metadata_engine.hpp"

namespace py = pybind11;
using namespace dkv;

PYBIND11_MODULE(native_sparse_metadata_engine, m) {
    m.doc() = "RCO-N: Native Sparse Metadata Engine — compact C++ metadata storage";

    m.attr("ENTRY_SIZE_BYTES") = sizeof(SparseMetadataEntry);

    py::class_<NativeSparseMetadataEngine>(m, "NativeSparseMetadataEngine")
        .def(py::init<int>(), py::arg("max_sessions") = 256)
        .def("create_session",   &NativeSparseMetadataEngine::create_session)
        .def("remove_session",   &NativeSparseMetadataEngine::remove_session)
        .def("has_session",      &NativeSparseMetadataEngine::has_session)
        .def("update",           &NativeSparseMetadataEngine::update,
             py::arg("session_id"),
             py::arg("sparse_ratio"), py::arg("confidence"), py::arg("continuity"),
             py::arg("zone_id"), py::arg("repair_type"),
             py::arg("sparse_safe"), py::arg("repair_pending"), py::arg("degraded"))
        .def("is_sparse_safe",   &NativeSparseMetadataEngine::is_sparse_safe)
        .def("get_confidence",   &NativeSparseMetadataEngine::get_confidence)
        .def("get_sparse_ratio", &NativeSparseMetadataEngine::get_sparse_ratio)
        .def("record_token",     &NativeSparseMetadataEngine::record_token,
             py::arg("session_id"), py::arg("count") = 1)
        .def("record_fusion",    &NativeSparseMetadataEngine::record_fusion)
        .def("get_sessions_below_confidence",
             &NativeSparseMetadataEngine::get_sessions_below_confidence)
        .def("get_sessions_needing_repair",
             &NativeSparseMetadataEngine::get_sessions_needing_repair)
        .def("session_count",    &NativeSparseMetadataEngine::session_count)
        .def("get_stats_json",   &NativeSparseMetadataEngine::get_stats_json);
}
