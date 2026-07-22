// bindings.cpp — pybind11 bindings for NativeDecodeScheduler
// RCO-N Phase 41.1

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "native_decode_scheduler.hpp"

namespace py = pybind11;
using namespace dkv;

PYBIND11_MODULE(native_decode_scheduler, m) {
    m.doc() = "RCO-N: Native Decode Scheduler — persistent batch management in C++";

    py::class_<SchedulerStats>(m, "SchedulerStats")
        .def_readonly("total_batch_steps",      &SchedulerStats::total_batch_steps)
        .def_readonly("total_tokens_scheduled", &SchedulerStats::total_tokens_scheduled)
        .def_readonly("slot_fills",             &SchedulerStats::slot_fills)
        .def_readonly("slot_evictions",         &SchedulerStats::slot_evictions)
        .def_readonly("starvation_events",      &SchedulerStats::starvation_events)
        .def_readonly("avg_batch_size",         &SchedulerStats::avg_batch_size)
        .def_readonly("avg_starvation_gap_ms",  &SchedulerStats::avg_starvation_gap_ms)
        .def_readonly("max_starvation_gap_ms",  &SchedulerStats::max_starvation_gap_ms)
        .def_readonly("scheduler_overhead_us",  &SchedulerStats::scheduler_overhead_us)
        .def_readonly("active_slots",           &SchedulerStats::active_slots)
        .def_readonly("admission_queue_depth",  &SchedulerStats::admission_queue_depth);

    py::class_<NativeDecodeScheduler>(m, "NativeDecodeScheduler")
        .def(py::init<int, double>(),
             py::arg("max_batch_size") = 32,
             py::arg("starvation_threshold_ms") = 1.5)
        .def("admit",         &NativeDecodeScheduler::admit,
             py::arg("session_id"), py::arg("request_id"),
             py::arg("max_tokens") = 128, py::arg("priority") = 0)
        .def("complete",      &NativeDecodeScheduler::complete)
        .def("cancel",        &NativeDecodeScheduler::cancel)
        .def("prepare_batch", &NativeDecodeScheduler::prepare_batch,
             py::return_value_policy::move)
        .def("record_token",  &NativeDecodeScheduler::record_token,
             py::arg("session_id"), py::arg("count") = 1)
        .def("step_begin",    &NativeDecodeScheduler::step_begin)
        .def("step_end",      &NativeDecodeScheduler::step_end)
        .def("get_stats",     &NativeDecodeScheduler::get_stats)
        .def("get_stats_json",&NativeDecodeScheduler::get_stats_json)
        .def("max_batch_size",&NativeDecodeScheduler::max_batch_size)
        .def("set_max_batch_size", &NativeDecodeScheduler::set_max_batch_size);
}
