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
    pred_0 = loss_function->link_function(1.5*y.array().mean());
    //pred_0 = 0;
    
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    dVector g = loss_function->dloss(y, pred ); //dVector::Zero(n1,1)
    dVector h = loss_function->ddloss(y, pred ); //dVector::Zero(n1,1)
    root = update_tree_info(X, y,g,h, root,0);

    //root = update_tree_info(X, y, root,0);
} 