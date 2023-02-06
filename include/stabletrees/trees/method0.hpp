#pragma once
#include "tree.hpp"
#include "node.hpp"


class Method0: public Tree{
    public:
        Method0(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity);
        Method0();
        virtual void update(dMatrix &X, dVector &y);
};

Method0::Method0():Tree(){
    Tree();
}

Method0::Method0(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity);
}



void Method0::update(dMatrix &X, dVector &y){
    if(root == NULL){
        learn(X,y);
    }
    root = update_tree_info(X, y, root,0);
} 