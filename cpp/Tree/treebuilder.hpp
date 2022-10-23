#pragma once
#ifndef __TREEBUILDER_HPP_INCLUDED__

#define __TREEBUILDER_HPP_INCLUDED__

#include <Eigen/Dense>
#include "node.hpp"
#include "splitter.hpp"


using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;


using namespace std;



class Treebuilder{

    public:
        int variable;
        Treebuilder(); 
        bool all_same(dVector &vec);
        bool all_same_features_values(dMatrix  &X);
        Node* build_tree(dMatrix  &X, dVector &y);
        tuple<dVector, dVector> get_masks(dVector &feature, dVector &y, double value);


};

Treebuilder::Treebuilder(){
    this->variable = 1;
} 

bool Treebuilder::all_same(dVector &vec){
    bool same = true;
    for(int i=0; i< vec.rows(); i++){
        if(vec[i]!=vec[0] ){
            same=false;
            break;
        }
    }
    return same;
}

bool Treebuilder::all_same_features_values(dMatrix &X){
    bool same = true;
    dVector feature;
    for(int i =0; i<X.cols(); i++){
        feature = X.col(i);
        if(!all_same(feature)){
            same=false;
            break;
        }
    }
    return same;
}

tuple<dVector, dVector> Treebuilder::get_masks(dVector &feature, dVector &y, double value){
    std::vector<double> left_values;
    std::vector<double> right_values;
    for(int i=0; i<y.rows();i++){
        if(feature[i]<=value){
            left_values.push_back(y[i]);
        }else{
            right_values.push_back(y[i]);
        }
    }
    dVector left_values_v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(left_values.data(), left_values.size());
    dVector right_values_v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(right_values.data(), right_values.size());

    return tuple<dVector, dVector> (left_values_v, right_values_v);
}


/*
Node* Treebuilder::build_tree(dMatrix  &X, dVector &y){
    double score;
    double split_value;
    np.ndarray[np.npy_bool, ndim=1] mask_left;
    np.ndarray[np.npy_bool, ndim=1] mask_right;
    Node* node;
}
*/
#endif