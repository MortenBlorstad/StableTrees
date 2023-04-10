#pragma once
#include "tree.hpp"


class StabilityRegularization: public Tree{
    public:
        StabilityRegularization(double gamma, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf,bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state);
        StabilityRegularization();
        virtual void update(dMatrix &X, dVector &y);
    private:
        double gamma;
        
};

StabilityRegularization::StabilityRegularization():Tree(){
    Tree();
    gamma = 0.5;
}

StabilityRegularization::StabilityRegularization(double gamma, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    this->gamma = gamma;
}

void StabilityRegularization::update(dMatrix &X, dVector &y){
     if(this->root == NULL){
        this->learn(X,y);
    }else{
        dVector ypred1 = loss_function->link_function(this->predict(X));

        // dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1), ypred1, lambda); 
        // dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1), ypred1, lambda);

        double original_mean = y.array().mean() +y.array().mean()/2 ;

        // // a quick fix for SL, since for some updates some of the prediction become extremely large (inf). fix by unsuring log lambda is is at least 0.
        // if(_criterion ==1){ // if poisson loss,
        //     original_mean = max(original_mean,exp(1)); // one need to ensure that $\bar{y}$ is sufficiently large,
        //                                             // so that the Poisson distribution can be approximated by a normal distribution with mean $\lambda$ and variance $\lambda$.
        //                                              //  If $\bar{y}$ is small, then the Poisson distribution is better approximated by a gamma distribution.
        // }
        
        pred_0 = loss_function->link_function(original_mean);

        dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
        dVector g = loss_function->dloss(y, pred, ypred1, gamma); 
        dVector h = loss_function->ddloss(y, pred, ypred1, gamma);

        splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity, max_features, learning_rate);
        this->root = build_tree(X, y, g, h, 0,this->root );
    }     
}





