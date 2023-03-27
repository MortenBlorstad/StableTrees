#pragma once
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"


class GBT{
    public:
        explicit GBT(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,double learning_rate);
        explicit GBT();
        void update(dMatrix &X, dVector &y);
        void learn(dMatrix &X, dVector &y);
        dVector predict(dMatrix &X);
        Tree* first_tree;
        Tree* current_tree;
        Tree* new_tree;
    protected:
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        double learning_rate;
        int n_estimator;
        double initial_pred;
        unsigned int random_state;
};

GBT::GBT(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,double learning_rate){
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }
    this-> min_split_sample = min_split_sample;
    this->min_samples_leaf = min_samples_leaf;
    this->_criterion = _criterion;
    this->adaptive_complexity = adaptive_complexity;
    this->learning_rate = learning_rate;
    this->n_estimator = n_estimator;
    this->random_state = 1;
}

void GBT::learn(dMatrix &X, dVector &y){
    LossFunction* loss_function = new LossFunction(_criterion);
    initial_pred = loss_function->link_function(y.array().mean());
    dVector pred = dVector::Zero(y.size());
    pred.setConstant(initial_pred);

    dVector g = loss_function->dloss(y,  pred);
    dVector h = loss_function->ddloss(y, pred); 

    this->first_tree = new Tree(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX,learning_rate,this->random_state);
    first_tree->learn_difference(X,y,g,h);
    pred = pred + first_tree->learning_rate* first_tree->predict(X);
    Tree* current_tree = this->first_tree;
    for(int i = 0; i<n_estimator;i++){
        g = loss_function->dloss(y,  pred);
        h = loss_function->ddloss(y,  pred);
        Tree* new_tree = new Tree(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX, learning_rate,this->random_state);
        new_tree->learn_difference(X,y,g,h);
        pred = pred + new_tree->learning_rate*new_tree->predict(X);
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
    }


}

dVector GBT::predict(dMatrix &X){
    Tree* current_tree = this->first_tree;
    dVector pred =  dVector::Zero(X.rows());
    pred.setConstant(initial_pred);
    while(current_tree !=NULL){
        pred = pred + current_tree->learning_rate*current_tree->predict(X);
        current_tree = current_tree->next_tree;
    }
    return pred;
}

void GBT::update(dMatrix &X, dVector &y){
    LossFunction* loss_function = new LossFunction(_criterion);
    initial_pred = loss_function->link_function(y.array().mean());
    dVector pred = dVector::Zero(y.size());
    pred.setConstant(initial_pred);
    dVector pred1 = this->predict(X);
    dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1), pred, 0.25); 
    dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1), pred, 0.25);

    this->first_tree = new Tree(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX,learning_rate,this->random_state);
    first_tree->learn_difference(X,y,g,h);
    pred = pred + first_tree->learning_rate* first_tree->predict(X);
    Tree* current_tree = this->first_tree;
    for(int i = 0; i<n_estimator;i++){
        g = loss_function->dloss(y, pred, pred1, 0.25); 
        h = loss_function->ddloss(y, pred, pred1, 0.25);
        Tree* new_tree = new Tree(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX, learning_rate,this->random_state);
        new_tree->learn_difference(X,y,g,h);
        pred = pred + new_tree->learning_rate*new_tree->predict(X);
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
    }

    


   
}