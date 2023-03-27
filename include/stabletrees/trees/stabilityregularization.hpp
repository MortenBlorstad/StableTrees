#pragma once
#include "tree.hpp"


class StabilityRegularization: public Tree{
    public:
        StabilityRegularization(double lambda, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf,bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state);
        StabilityRegularization();
        virtual void update(dMatrix &X, dVector &y);
    private:
        double lambda;
        
};

StabilityRegularization::StabilityRegularization():Tree(){
    Tree();
    lambda = 0.5;
}

StabilityRegularization::StabilityRegularization(double lambda, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    this->lambda = lambda;
}

void StabilityRegularization::update(dMatrix &X, dVector &y){
     if(this->root == NULL){
        this->learn(X,y);
    }else{
        dVector ypred1 = this->predict(X);
        dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1), ypred1, lambda); 
        dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1), ypred1, lambda);
        splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity, max_features, learning_rate);
        this->root = build_tree(X, y, g, h, 0,this->root );
    }     
}





