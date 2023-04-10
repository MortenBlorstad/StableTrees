
#include <pybind11\pybind11.h>
#include <pybind11\eigen.h>
#include <cstdio>


using namespace std;


#include <pybind11/stl.h>

#include "node.hpp"
#include "splitters\splitter.hpp"
#include "criterions\criterion.hpp"
#include "trees\tree.hpp"
#include "trees\newtree.hpp"
#include "trees\abutree.hpp"
#include "trees\abutreeI.hpp"
#include "criterions\MSE.hpp"
#include "criterions\Poisson.hpp"
#include "optimism\cir.hpp"
#include "trees\naiveupdate.hpp"
#include "trees\stabilityregularization.hpp"
#include "trees\treereevaluation.hpp"
#include "ensemble\GBT.hpp"
#include "ensemble\randomforest.hpp"





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
        .def("toString", &Node::toString)
        .def("get_features_indices", &Node::get_features_indices);

    
    py::class_<Tree>(m, "Tree")
        .def(py::init<int, int , double,int, bool, int ,double, unsigned int>())
        .def("all_same", &Tree::all_same)
        .def("all_same_features_values", &Tree::all_same_features_values )
        .def("get_masks", &Tree::get_masks)
        .def("build_tree", &Tree::build_tree)
        .def("learn", &Tree::learn)
        .def("learn_difference", &Tree::learn_difference)
        .def("get_root", &Tree::get_root)
        .def("predict", &Tree::predict)
        .def("predict_uncertainty", &Tree::predict_uncertainty)
        .def("update", &Tree::update)
        .def("make_node_list", &Tree::make_node_list);
    
    
    
     py::class_<NaiveUpdate>(m, "NaiveUpdate")
        .def(py::init<int, int, double,int, bool, int,double,unsigned int>())
            .def("learn", &NaiveUpdate::learn)
            .def("predict", &NaiveUpdate::predict)
            .def("update", &NaiveUpdate::update)
            .def("get_root", &NaiveUpdate::get_root);
        
    py::class_<TreeReevaluation>(m, "TreeReevaluation")
        .def(py::init<double, double,int, int, double,int, bool, int,double,unsigned int>())
            .def("learn", &TreeReevaluation::learn)
            .def("predict", &TreeReevaluation::predict)
            .def("update", &TreeReevaluation::update)
            .def("get_root", &TreeReevaluation::get_root)
            .def("get_mse_ratio", &TreeReevaluation::get_mse_ratio)
            .def("get_eps", &TreeReevaluation::get_eps)
            .def("get_obs", &TreeReevaluation::get_obs);



    py::class_<StabilityRegularization>(m, "StabilityRegularization")
         .def(py::init<double, int, int, double, int,bool, int,double,unsigned int>())
            .def("learn", &StabilityRegularization::learn)
            .def("predict", &StabilityRegularization::predict)
            .def("update", &StabilityRegularization::update)
            .def("get_root", &StabilityRegularization::get_root);
        

    py::class_<AbuTreeI>(m, "AbuTreeI")
        .def(py::init<int, int, double,int,bool, int, double,unsigned int>())
            .def("learn", &AbuTreeI::learn)
            .def("predict", &AbuTreeI::predict)
            .def("update", &AbuTreeI::update)
            .def("predict_uncertainty", &AbuTreeI::predict_uncertainty)
            .def("predict_info", &AbuTreeI::predict_info)
            .def("get_root", &AbuTreeI::get_root);

    py::class_<AbuTree>(m, "AbuTree")
        .def(py::init<int, int, double,int,bool, int,double,unsigned int>())
            .def("learn", &AbuTree::learn)
            .def("predict", &AbuTree::predict)
            .def("update", &AbuTree::update)
            .def("get_root", &AbuTree::get_root);

    py::class_<GBT>(m, "GBT")
        .def(py::init<int,int, int , double,int, bool ,double>())
        .def("learn", &GBT::learn)
        .def("predict", &GBT::predict)
        .def("update", &GBT::update);
    

    py::class_<RandomForest>(m, "RandomForest")
    .def(py::init<int,int, int , double,int, bool ,int,int>())
    .def("learn", &RandomForest::learn)
    .def("predict", &RandomForest::predict)
    .def("update", &RandomForest::update);


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

    py::class_<MSEABU>(m, "MSEABU")
    .def(py::init<int>())
        .def("get_score", &MSEABU::get_score)
        .def("init", &MSEABU::init)
        .def("update", &MSEABU::update)
        .def("get_root", &MSEABU::reset)
        .def("node_impurity", &MSEABU::node_impurity)
        .def("reset", &MSEABU::reset);
    
    py::class_<PoissonABU>(m, "PoissonABU")
    .def(py::init<>())
        .def("get_score", &PoissonABU::get_score)
        .def("init", &PoissonABU::init)
        .def("update", &PoissonABU::update)
        .def("get_root", &PoissonABU::reset)
        .def("node_impurity", &PoissonABU::node_impurity)
        .def("reset", &PoissonABU::reset);


    py::class_<PoissonReg>(m, "PoissonReg")
        .def(py::init<>())
            .def("get_score", &PoissonReg::get_score)
            .def("init", &PoissonReg::init)
            .def("update", &PoissonReg::update)
            .def("get_root", &PoissonReg::reset)
            .def("node_impurity", &PoissonReg::node_impurity)
            .def("reset", &PoissonReg::reset);



    py::class_<Splitter>(m, "Splitter")
        .def(py::init<int, double,bool, int, double>())
            .def("find_best_split", &Splitter::find_best_split)
            .def("get_reduction", &Splitter::get_reduction);

    py::class_<NewTree>(m, "NewTree")
        .def(py::init<int, int , double,int, bool, int ,double, unsigned int>())
            .def("build_tree", &NewTree::build_tree)
            .def("learn", &NewTree::learn)
            .def("get_root", &NewTree::get_root)
            .def("predict", &NewTree::predict)
            .def("update", &NewTree::update);


    



    m.def("rnchisq", &rnchisq);
    m.def("cir_sim_vec",&cir_sim_vec);
    m.def("cir_sim_mat",&cir_sim_mat);
    


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}