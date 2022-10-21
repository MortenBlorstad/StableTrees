#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <cstdio>

#include "node.hpp"
namespace py = pybind11;

int add(int i, int j)
{
    printf("C++ being called! %d %d\n", i, j);
    return i + j;
}

PYBIND11_MODULE(stable_trees, m)
{
    m.doc() = "C++ KF implementation wrappers";  // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    py::class_<Node>(m, "Node")
        .def(py::init<>())
        .def("is_leaf", &Node::is_leaf)
        .def("get_right_node", &Node::get_right_node)
        .def("get_left_node", &Node::get_left_node);
}
