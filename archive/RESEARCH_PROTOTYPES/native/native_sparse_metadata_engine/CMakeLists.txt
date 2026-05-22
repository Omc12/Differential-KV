cmake_minimum_required(VERSION 3.18)
project(native_sparse_metadata_engine CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
endif()

pybind11_add_module(native_sparse_metadata_engine
    native_sparse_metadata_engine.cpp bindings.cpp)
target_include_directories(native_sparse_metadata_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    target_compile_options(native_sparse_metadata_engine PRIVATE /O2)
else()
    target_compile_options(native_sparse_metadata_engine PRIVATE -O3)
endif()
install(TARGETS native_sparse_metadata_engine DESTINATION .)
