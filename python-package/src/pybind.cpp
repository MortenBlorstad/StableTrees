
#include <pybind11\pybind11.h>
#include <pybind11\eigen.h>
#include <cstdio>


using namespace std;


#include <pybind11/stl.h>

#include "node.hpp"
#include "splitters\splitter.hpp"
#include "trees\approximate_bayesian_update.hpp"
#include "splitters\probabalisticsplitter.hpp"
#include "criterions\criterion.hpp"
#include "trees\tree.hpp"
#include "trees\abutree.hpp"
#include "trees\method2.hpp"
#include "trees\method0.hpp"
#include "trees\method1.hpp"
#include "trees\probabalistictree.hpp"
#include "trees\evotree.hpp"
#include "criterions\MSE.hpp"
#include "criterions\Poisson.hpp"
#include "optimism\cir.hpp"



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
        .def(py::init<double,double, double, int, int, double>())
        .def(py::init<double, int>())
        .def("is_leaf", &Node::is_leaf)
        .def("set_left_node", &Node::set_left_node)
        .def("set_right_node", &Node::set_right_node)
        .def("get_right_node", &Node::get_right_node)
        .def("get_left_node", &Node::get_left_node)
        .def("predict", &Node::predict)
        .def("nsamples", &Node::nsamples)
        .def("get_split_score", &Node::get_split_score)
        .def("get_impurity", &Node::get_impurity)
        .def("get_split_feature", &Node::get_split_feature)
        .def("get_split_value", &Node::get_split_value)
        .def("copy", &Node::copy)
        .def("toString", &Node::toString);

    
    py::class_<Tree>(m, "Tree")
        .def(py::init<int, int , double,int, bool >())
        .def("all_same", &Tree::all_same)
        .def("all_same_features_values", &Tree::all_same_features_values )
        .def("get_masks", &Tree::get_masks)
        .def("build_tree", &Tree::build_tree)
        .def("learn", &Tree::learn)
        .def("get_root", &Tree::get_root)
        .def("predict", &Tree::predict)
        .def("update", &Tree::update)
        .def("make_node_list", &Tree::make_node_list);
        
    py::class_<Method2>(m, "Method2")
        .def(py::init<double, int, int, double, int,bool>())
            .def("learn", &Method2::learn)
            .def("predict", &Method2::predict)
            .def("update", &Method2::update)
            .def("get_root", &Method2::get_root);

    py::class_<Method1>(m, "Method1")
        .def(py::init<int, int, double,int, bool>())
            .def("learn", &Method1::learn)
            .def("predict", &Method1::predict)
            .def("update", &Method1::update)
            .def("get_root", &Method1::get_root)
            .def("get_mse_ratio", &Method1::get_mse_ratio)
            .def("get_eps", &Method1::get_eps)
            .def("get_obs", &Method1::get_obs);

    py::class_<Method0>(m, "Method0")
        .def(py::init<int, int, double,int, bool>())
            .def("learn", &Method0::learn)
            .def("predict", &Method0::predict)
            .def("update", &Method0::update)
            .def("get_root", &Method0::get_root);

    py::class_<ProbabalisticTree>(m, "ProbabalisticTree")
        .def(py::init<int, int, double,int, bool, int>())
            .def("learn", &ProbabalisticTree::learn)
            .def("predict", &ProbabalisticTree::predict)
            .def("update", &ProbabalisticTree::update)
            .def("get_root", &ProbabalisticTree::get_root)
            .def("crossover", &ProbabalisticTree::crossover)
            .def("make_node_list", &ProbabalisticTree::make_node_list)
            .def("copy", &ProbabalisticTree::copy);

    py::class_<EvoTree>(m, "EvoTree")
        .def(py::init<int, int, double,int, bool, int>())
            .def("learn", &EvoTree::learn)
            .def("predict", &EvoTree::predict)
            .def("update", &EvoTree::update)
            .def("get_root", &EvoTree::get_root)
            .def("breed", &EvoTree::breed)
            .def("create_population", &EvoTree::create_population)
            .def("generate_population", &EvoTree::generate_population)
            .def("fitness_function", &EvoTree::fitness_function);

    py::class_<ABU>(m, "AbuTree")
        .def(py::init<int, int, double,int>())
            .def("learn", &ABU::learn)
            .def("predict", &ABU::predict)
            .def("update", &ABU::update)
            .def("get_root", &ABU::get_root);

    py::class_<AbuTree>(m, "AbuTreeI")
        .def(py::init<int, int, double,int>())
            .def("learn", &AbuTree::learn)
            .def("predict", &AbuTree::predict)
            .def("update", &AbuTree::update)
            .def("get_root", &AbuTree::get_root);


    py::class_<MSE>(m, "MSE")
        .def(py::init<>())
            .def("get_score", &MSE::get_score)
            .def("init", &MSE::init)
            .def("update", &MSE::update)
            .def("get_root", &MSE::reset)
            .def("node_impurity", &MSE::node_impurity)
            .def("reset", &MSE::reset);

    py::class_<Poisson>(m, "Poisson")
        .def(py::init<>())
            .def("get_score", &Poisson::get_score)
            .def("init", &Poisson::init)
            .def("update", &Poisson::update)
            .def("get_root", &Poisson::reset)
            .def("node_impurity", &Poisson::node_impurity)
            .def("reset", &Poisson::reset);

    py::class_<MSEReg>(m, "MSEReg")
        .def(py::init<>())
            .def("get_score", &MSEReg::get_score)
            .def("init", &MSEReg::init)
            .def("update", &MSEReg::update)
            .def("get_root", &MSEReg::reset)
            .def("node_impurity", &MSEReg::node_impurity)
            .def("reset", &MSEReg::reset);


    py::class_<PoissonReg>(m, "PoissonReg")
        .def(py::init<>())
            .def("get_score", &PoissonReg::get_score)
            .def("init", &PoissonReg::init)
            .def("update", &PoissonReg::update)
            .def("get_root", &PoissonReg::reset)
            .def("node_impurity", &PoissonReg::node_impurity)
            .def("reset", &PoissonReg::reset);



    py::class_<Splitter>(m, "Splitter")
        .def(py::init<int, double, int,bool>())
            .def("find_best_split", &Splitter::find_best_split)
            .def("select_split_from_all", &Splitter::select_split_from_all);


    py::class_<ProbabalisticSplitter>(m, "ProbabalisticSplitter")
        .def(py::init<int, double, int,bool,int>())
            .def("find_best_split", &ProbabalisticSplitter::find_best_split)
            .def("select_split_from_all", &ProbabalisticSplitter::select_split_from_all);


    m.def("rnchisq", &rnchisq);
    m.def("cir_sim_vec",&cir_sim_vec);
    m.def("cir_sim_mat",&cir_sim_mat);
    


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
