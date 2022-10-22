#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <cstdio>

#include "node.hpp"
namespace py = pybind11;



PYBIND11_MODULE(stable_trees, m)
{
    m.doc() = "C++ KF implementation wrappers";  // optional module docstring
    
    py::class_<Node>(m, "Node")
        .def(py::init<double,double, int, int, double>())
        .def(py::init<double, int>())
        .def("is_leaf", &Node::is_leaf)
        .def("set_left_node", &Node::set_left_node)
        .def("set_right_node", &Node::set_right_node)
        .def("get_right_node", &Node::get_right_node)
        .def("get_left_node", &Node::get_left_node)
        .def("predict", &Node::predict);
}
