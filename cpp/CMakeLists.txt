project(StableTrees)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(thirdparty/pybind11)
include_directories(thirdparty/eigen)

pybind11_add_module(stable_trees wrappers.cpp node.hpp)
