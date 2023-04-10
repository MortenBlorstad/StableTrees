#pragma once
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include "abusplitter.hpp"

class AbuTreeI: public Tree{
    public:
        AbuTreeI(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state);
        AbuTreeI();
        virtual void update(dMatrix &X, dVector &y);
        dMatrix predict_info(dMatrix &X);
        tuple<bool,int,double, double,double,double,double,double>  AbuTreeI::find_update_split(const dMatrix &X,const dVector &y, const dVector &g,const dVector &h,const dVector &weights);
    private:
        dVector predict_info_obs(dVector  &obs);
        dMatrix sample_X(const dMatrix &X, int n1);
        int bootstrap_seed ;
};

AbuTreeI::AbuTreeI():Tree(){
    Tree(); 
    bootstrap_seed=0;
}

AbuTreeI::AbuTreeI(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double learning_rate, unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    bootstrap_seed=0;
}


dVector AbuTreeI::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(2,1);
    while(node !=NULL){
        if(node->is_leaf()){
            info(0,1) = node->predict();
            if(std::isnan(node->y_var)||std::isnan(node->w_var) || std::isnan((node->y_var/node->w_var)/node->n_samples) ){
                    std::cout << "y_var or w_var contains NaN:" << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(node->y_var< 0 || node->w_var <0 || (node->y_var/node->w_var)/node->n_samples<0){
                    std::cout << "y_var or w_var <0: " << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(node->w_var <=0){
                node->w_var =0.00001;
            }
            if(node->y_var <=0){
                node->y_var =0.00001;
            }
            if(_criterion ==1){ //poisson only uses prediction variance
                //info(1,1) = (node->y_var/node->w_var)/node->n_samples;
                info(1,1) = 1/(node->w_var/node->n_samples);
            }
            else{ //mse uses both response and prediction variance
                
                info(1,1) = (node->y_var/node->w_var)/node->n_samples;
            }
            return info;
        }else{
            if(obs(node->split_feature) <= node->split_value){
                node = node->left_child;
            }else{
                node = node->right_child;
            }
        }
    }
}
dMatrix AbuTreeI::predict_info(dMatrix &X){
    int n = X.rows();
    dMatrix leaf_info(n,2);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        dVector info =predict_info_obs(obs);
        for (size_t j = 0; j < info.size(); j++)
        {
            leaf_info(i,j) = info(j);
        }
    }
    return leaf_info;
}


void AbuTreeI::update(dMatrix &X, dVector &y){
    //printf("%d\n", n1);
    dMatrix Xb = sample_X(X,n1);
    dMatrix info = predict_info(Xb);
    dVector weights = info.col(1);//array().min(1000).max(0);
    dVector yb = info.col(0);
    // for (size_t i = 0; i < yb.size(); i++)
    // {
    //     printf("yb = %f \n", yb(i));
    // }

    // complete the squares 
    dVector hb = 2*weights.array();
    dVector gb = -1*hb.array().cwiseProduct(yb.array());
    dVector indicator_b = dVector::Constant(yb.size(),0,  0) ;
    yb = yb.array()+pred_0;

    pred_0 = loss_function->link_function(y.array().mean());//
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    // for (size_t i = 0; i < pred.size(); i++)
    // {
    //     printf("pred = %f \n", pred(i));
    // }

    dVector g = loss_function->dloss(y, pred ); 
    dVector h = loss_function->ddloss(y, pred );
    dVector gamma = dVector::Constant(y.size(),0,  0) ;
    dVector indicator = dVector::Constant(y.size(),0,  1) ;
    
    // dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1)); 
    // dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1) );
    dMatrix X_concat(X.rows()+Xb.rows(), X.cols());
    dVector y_concat(y.rows()+yb.rows(), 1);
    dVector g_concat(g.rows() + gb.rows(), 1); 
    dVector h_concat(h.rows() + hb.rows(), 1); 

    dVector gamma_concat(y.rows() + yb.rows(), 1); 
    dVector indicator_concat(y.rows() + yb.rows(), 1);
    

     for (int i = 0; i < weights.size(); i++) {
        if (std::isnan(weights(i)) || weights(i)<=0) {
            std::cout << "weights contains NaN at index " << i <<" - "<< weights(i) << std::endl;
        }
    }


    g_concat <<g,gb ;
    h_concat <<h,hb;
    X_concat <<X,Xb;
    y_concat <<y, loss_function->inverse_link_function(yb.array());
    gamma_concat <<gamma, weights;
    indicator_concat <<indicator, indicator_b;
    
  
    total_obs = X_concat.rows();
    splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity, max_features,learning_rate);

    this->root = build_tree(X_concat, y_concat, g_concat, h_concat, 0,this->root,indicator_concat, gamma_concat);
    n1 = total_obs;
}


dMatrix AbuTreeI::sample_X(const dMatrix &X, int n1){
    std::mt19937 gen(bootstrap_seed);
    std::uniform_int_distribution<size_t>  distr(0, X.rows()-1);
    dMatrix X_sample(n1, X.cols());
    for (size_t i = 0; i < n1; i++)
    {   
        size_t ind = distr(gen);
        for (size_t j = 0; j < X.cols(); j++)
        {   
            double x_b = X(ind,j);
            X_sample(i,j) = x_b;
        } 
    }
    bootstrap_seed+=1;
    return X_sample;
}




