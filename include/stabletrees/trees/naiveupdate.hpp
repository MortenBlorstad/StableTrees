#pragma once
#include "tree.hpp"
#include "node.hpp"


class NaiveUpdate: public Tree{
    public:
        NaiveUpdate(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state);
        NaiveUpdate();
        virtual void update(dMatrix &X, dVector &y);
};

NaiveUpdate::NaiveUpdate():Tree(){
    Tree();
}

NaiveUpdate::NaiveUpdate(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,double learning_rate,unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
}



void NaiveUpdate::update(dMatrix &X, dVector &y){
    if(root == NULL){
        learn(X,y);
    }
    root = update_tree_info(X, y, root,0);
} 