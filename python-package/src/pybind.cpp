
#include <pybind11\pybind11.h>
#include <pybind11\eigen.h>
#include <cstdio>


using namespace std;


#include <pybind11/stl.h>

#include "node.hpp"
#include "splitters\splitter.hpp"
#include "criterions\criterion.hpp"
#include "trees\tree.hpp"
#include "trees\method2.hpp"
#include "criterions\MSE.hpp"
#include "criterions\Poisson.hpp"


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(_stabletrees, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: stable_trees
        .. autosummary::
           :toctree: _generate

           get_predictions

    )pbdoc";

    py::class_<Node>(m, "Node")
        .def(py::init<double,double, int, int, double>())
        .def(py::init<double, int>())
        .def("is_leaf", &Node::is_leaf)
        .def("set_left_node", &Node::set_left_node)
        .def("set_right_node", &Node::set_right_node)
        .def("get_right_node", &Node::get_right_node)
        .def("get_left_node", &Node::get_left_node)
        .def("predict", &Node::predict)
        .def("text", &Node::text)
        .def("nsamples", &Node::nsamples)
        .def("get_split_score", &Node::get_split_score);

    
    py::class_<Tree>(m, "Tree")
        .def(py::init<int , double >())
        .def("all_same", &Tree::all_same)
        .def("all_same_features_values", &Tree::all_same_features_values )
        .def("get_masks", &Tree::get_masks)
        .def("build_tree", &Tree::build_tree)
        .def("learn", &Tree::learn)
        .def("get_root", &Tree::get_root)
        .def("example", &Tree::example)
        .def("predict", &Tree::predict)
        .def("update", &Tree::update);

    py::class_<Method2>(m, "Method2")
        .def(py::init<int, double>())
            .def("learn", &Method2::learn)
            .def("predict", &Method2::predict)
            .def("update", &Method2::update)
            .def("get_root", &Method2::get_root);

    py::class_<MSE>(m, "MSE")
        .def(py::init<>())
            .def("get_score", &MSE::get_score)
            .def("init", &MSE::init)
            .def("update", &MSE::update)
            .def("get_root", &MSE::reset)
            .def("node_impurity", &MSE::node_impurity);

    py::class_<Poisson>(m, "Poisson")
        .def(py::init<>())
            .def("get_score", &Poisson::get_score)
            .def("init", &Poisson::init)
            .def("update", &Poisson::update)
            .def("get_root", &Poisson::reset)
            .def("node_impurity", &Poisson::node_impurity);

    py::class_<MSEReg>(m, "MSEReg")
        .def(py::init<>())
            .def("get_score", &MSEReg::get_score)
            .def("init", &MSEReg::init)
            .def("update", &MSEReg::update)
            .def("get_root", &MSEReg::reset)
            .def("node_impurity", &MSEReg::node_impurity);


    py::class_<PoissonReg>(m, "PoissonReg")
        .def(py::init<>())
            .def("get_score", &PoissonReg::get_score)
            .def("init", &PoissonReg::init)
            .def("update", &PoissonReg::update)
            .def("get_root", &PoissonReg::reset)
            .def("node_impurity", &PoissonReg::node_impurity);
    


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
