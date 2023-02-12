#pragma once
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"

class AbuTreeI: public Tree{
    public:
        AbuTreeI(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity);
        AbuTreeI();
        virtual void update(dMatrix &X, dVector &y);
        dMatrix predict_info(dMatrix &X);
    private:
        dVector predict_info_obs(dVector  &obs);
        tuple<dMatrix,dVector> sample_X_y(const dMatrix &X,const dVector &y, int n1);
        dMatrix sample_X(const dMatrix &X, int n1);
};

AbuTreeI::AbuTreeI():Tree(){
    Tree(); 
}

AbuTreeI::AbuTreeI(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity);
}


dVector AbuTreeI::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(2,1);
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            info(0,1) = node->predict();
            if(_criterion ==1){
                info(1,1) = 1/(node->w_var/node->n_samples);
            }
            else{
                info(1,1) = (node->y_var/node->w_var)/node->n_samples;
            }
                
            //printf("asdas %f %f, %f ,%d \n", info(1,1),node->w_var, node->y_var, node->n_samples);
            return info;
        }else{
            //printf("feature %d, value %f, obs %f \n", node->split_feature, node->split_value,obs(node->split_feature));
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
    splitter = new Splitter(min_samples_leaf,total_obs,_criterion, adaptive_complexity);
    dMatrix Xb = sample_X(X,n1);
    dMatrix info = predict_info(Xb);
    dVector weights = info.col(1);
    dVector yb = info.col(0);
    
    dVector hb = 2*weights.array();

    dVector gb = -1*hb.array().cwiseProduct(yb.array());

    dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1)); 
    dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1) );
    dMatrix X_concat(X.rows()+Xb.rows(), X.cols());
    dVector y_concat(y.rows()+yb.rows(), 1);
    dVector g_concat(g.rows() + gb.rows(), 1); 
    dVector h_concat(h.rows() + hb.rows(), 1); 
    
    
    g_concat <<g,gb ;
    h_concat <<h,hb;
    X_concat <<X,Xb;
    y_concat <<y,yb;
    printf("%f\n", yb.array().sum());
    total_obs = X_concat.rows();
    this->root = build_tree(X_concat, y_concat, g_concat, h_concat, 0);
    n1 = total_obs;
}

tuple<dMatrix,dVector> AbuTreeI::sample_X_y(const dMatrix &X,const dVector &y, int n1){
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t>  distr(0, X.rows());
    dMatrix X_sample(n1, X.cols());
    dVector y_sample(n1,1);
    for (size_t i = 0; i < n1; i++)
    {
        size_t ind = distr(gen);
        for(size_t j =0; j<= X.cols(); j++){
            X_sample(i,j) = X(ind,j);
        }
        y_sample(i,0) = y(ind,0);
    }   
    return tuple<dMatrix,dVector>(X_sample,y_sample);
}

dMatrix AbuTreeI::sample_X(const dMatrix &X, int n1){
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t>  distr(0, X.rows());
    dMatrix X_sample(n1, X.cols());
    for (size_t i = 0; i < n1; i++)
    {   
        size_t ind = distr(gen);
        for (size_t j = 0; j < X.cols(); j++)
        {
            X_sample(i,j) = X(ind,j);
        } 
    }
    
    return X_sample;
}









