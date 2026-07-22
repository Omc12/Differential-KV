cmake_minimum_required(VERSION 3.20)
project(dkv_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/compressor_thread.cpp
    src/paging_stream.cu
    src/bindings.cpp
)

pybind11_add_module(dkv_core ${SOURCES})

target_include_directories(dkv_core PRIVATE include)

target_link_libraries(dkv_core PRIVATE
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
)

# Ensure deterministic CUDA compilation
target_compile_options(dkv_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

set_target_properties(dkv_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
