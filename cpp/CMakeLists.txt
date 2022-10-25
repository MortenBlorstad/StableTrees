project(StableTrees)
option(ENABLE_OPENMP "Use OpenMP parallelization" ON)
find_package(Python COMPONENTS Interpreter Development)


cmake_minimum_required(VERSION 3.10)
add_compile_options(/openmp, /openmp:llvm )

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)



pybind11_add_module(stable_trees wrappers.cpp Tree/node.hpp Tree/splitter.hpp Tree/tree.hpp)
