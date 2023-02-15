#pragma once
#include "tree.hpp"
#include "node.hpp"


class NaiveUpdate: public Tree{
    public:
        NaiveUpdate(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity);
        NaiveUpdate();
        virtual void update(dMatrix &X, dVector &y);
};

NaiveUpdate::NaiveUpdate():Tree(){
    Tree();
}

NaiveUpdate::NaiveUpdate(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity);
}



void NaiveUpdate::update(dMatrix &X, dVector &y){
    if(root == NULL){
        learn(X,y);
    }
    root = update_tree_info(X, y, root,0);
} 