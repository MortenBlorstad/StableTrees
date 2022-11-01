
#include <pybind11\pybind11.h>
#include <pybind11\eigen.h>
#include <cstdio>


using namespace std;
#include <pybind11/stl.h>

#include "Tree/node.hpp"
#include "Tree/splitter.hpp"
#include "Tree/tree.hpp"
#include "Tree/stabletree.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(stable_trees, m)
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

    py::class_<Splitter>(m, "Splitter")
        .def(py::init<>())
        .def("get_predictions", &Splitter::get_predictions,
                            "feature"_a,"y"_a, "value"_a )
        .def("sum_squared_error", &Splitter::sum_squared_error,
                            "y_true"_a, "y_pred"_a )
        .def("mse_criterion", &Splitter::mse_criterion)
        .def("find_best_split", &Splitter::find_best_split)
        .def("select_split", &Splitter::select_split)
        .def("para", &Splitter::para)
        .def("seq", &Splitter::seq);

    
    py::class_<Tree>(m, "Tree")
        .def(py::init<>())
        .def("all_same", &Tree::all_same)
        .def("all_same_features_values", &Tree::all_same_features_values )
        .def("get_masks", &Tree::get_masks)
        .def("build_tree", &Tree::build_tree)
        .def("learn", &Tree::learn)
        .def("get_root", &Tree::get_root)
        .def("example", &Tree::example)
        .def("predict", &Tree::predict)
        .def("update", &Tree::update);

    py::class_<StableTree>(m, "StableTree")
        .def(py::init<>())
        .def("all_same", &StableTree::all_same)
        .def("all_same_features_values", &Tree::all_same_features_values )
        .def("get_masks", &StableTree::get_masks)
        .def("build_tree", &StableTree::build_tree)
        .def("learn", &StableTree::learn)
        .def("get_root", &StableTree::get_root)
        .def("example", &StableTree::example)
        .def("update", &StableTree::update);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
