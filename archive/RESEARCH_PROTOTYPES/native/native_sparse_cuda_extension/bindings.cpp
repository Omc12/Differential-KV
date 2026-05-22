// SKO Phase 41.3: Native Sparse CUDA Extension Layer Pybind11 Bindings

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "native_sparse_cuda_extension.hpp"

namespace py = pybind11;

PYBIND11_MODULE(native_sparse_cuda_extension, m) {
    m.doc() = "Differential KV SKO Phase Native Sparse CUDA Extension";

    py::class_<diffkv::NativeSparseCudaExtension>(m, "NativeSparseCudaExtension")
        .def(py::init<>())
        .def("pack_gpu_metadata", &diffkv::NativeSparseCudaExtension::pack_gpu_metadata)
        .def("traverse_sparse_attention_blocks", &diffkv::NativeSparseCudaExtension::traverse_sparse_attention_blocks)
        .def("index_sparse_blocks", &diffkv::NativeSparseCudaExtension::index_sparse_blocks);
}
