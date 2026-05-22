// bindings.cpp — pybind11 for NativeTelemetryCounterLayer
// RCO-N Phase 41.1

#include <pybind11/pybind11.h>
#include "native_telemetry_counter_layer.hpp"

namespace py = pybind11;
using namespace diffkv;

PYBIND11_MODULE(native_telemetry_counter_layer, m) {
    m.doc() = "RCO-N: Native Telemetry Counter Layer — lock-free atomic counters";

    py::class_<NativeTelemetryCounterLayer>(m, "NativeTelemetryCounterLayer")
        .def(py::init<>())
        // Hot-path increment methods (GIL-released for max performance)
        .def("gpu_kernel_dispatched",  &NativeTelemetryCounterLayer::gpu_kernel_dispatched)
        .def("gpu_starvation_event",   &NativeTelemetryCounterLayer::gpu_starvation_event)
        .def("gpu_sync_stall",         &NativeTelemetryCounterLayer::gpu_sync_stall)
        .def("scheduler_step",         &NativeTelemetryCounterLayer::scheduler_step)
        .def("scheduler_admission",    &NativeTelemetryCounterLayer::scheduler_admission)
        .def("scheduler_eviction",     &NativeTelemetryCounterLayer::scheduler_eviction)
        .def("queue_enqueue",          &NativeTelemetryCounterLayer::queue_enqueue)
        .def("queue_dequeue",          &NativeTelemetryCounterLayer::queue_dequeue)
        .def("queue_cancel",           &NativeTelemetryCounterLayer::queue_cancel)
        .def("queue_reconnect",        &NativeTelemetryCounterLayer::queue_reconnect)
        .def("token_generated",        &NativeTelemetryCounterLayer::token_generated,
             py::arg("n") = 1)
        .def("governance_fired",       &NativeTelemetryCounterLayer::governance_fired)
        .def("governance_skipped",     &NativeTelemetryCounterLayer::governance_skipped)
        .def("dense_fallback",         &NativeTelemetryCounterLayer::dense_fallback)
        .def("partial_repair",         &NativeTelemetryCounterLayer::partial_repair)
        .def("fusion_call",            &NativeTelemetryCounterLayer::fusion_call)
        .def("telemetry_suppressed",   &NativeTelemetryCounterLayer::telemetry_suppressed)
        .def("telemetry_emitted",      &NativeTelemetryCounterLayer::telemetry_emitted)
        // Snapshot reads
        .def("get_snapshot_json",      &NativeTelemetryCounterLayer::get_snapshot_json)
        .def("governance_collapse_ratio",       &NativeTelemetryCounterLayer::governance_collapse_ratio)
        .def("queue_reconnect_coalesce_ratio",  &NativeTelemetryCounterLayer::queue_reconnect_coalesce_ratio)
        .def("telemetry_suppression_ratio",     &NativeTelemetryCounterLayer::telemetry_suppression_ratio)
        .def("reset_counters",         &NativeTelemetryCounterLayer::reset_counters);
}
