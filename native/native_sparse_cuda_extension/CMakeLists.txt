# SKO Phase 41.3: Native Sparse CUDA Extension Layer
# CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(native_sparse_cuda_extension)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    # Fallback: use pip-installed pybind11
    execute_process(
        COMMAND python -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
    )
    if(PYBIND11_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()
pybind11_add_module(native_sparse_cuda_extension
    native_sparse_cuda_extension.cpp
    bindings.cpp
)

target_compile_features(native_sparse_cuda_extension PRIVATE cxx_std_17)

# We optimize for speed
if(MSVC)
    target_compile_options(native_sparse_cuda_extension PRIVATE /O2 /fp:fast)
else()
    target_compile_options(native_sparse_cuda_extension PRIVATE -O3 -ffast-math)
endif()
