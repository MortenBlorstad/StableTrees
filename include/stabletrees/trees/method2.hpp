#pragma once
#include "tree.hpp"
#include "splitterReg.hpp"
class Method2: public Tree{
    public:
        Method2(double lambda, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf,bool adaptive_complexity);
        Method2();
        virtual void update(dMatrix &X, dVector &y);
        tuple<bool, int,double, double,double,double> find_split_update(dMatrix &X, dVector &y, dVector &yprev);
        
    private:
        Node* update_tree(dMatrix  &X, dVector &y, int depth, dVector &yprev);
        double lambda;
        SplitterReg* update_splitter;

};

Method2::Method2():Tree(){
    Tree();
    lambda = 0.5;
}

Method2::Method2(double lambda, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity);
    this->lambda = lambda;
}

tuple<bool, int,double, double,double,double> Method2::find_split_update(dMatrix &X, dVector &y, dVector &yprev){
    return update_splitter->find_best_split(X, y, yprev);
}

void Method2::update(dMatrix &X, dVector &y) {
    update_splitter = new SplitterReg(lambda,min_samples_leaf,total_obs,_criterion,adaptive_complexity);
    if(adaptive_complexity){
        update_splitter->cir_sim = cir_sim;
        update_splitter->grid_end = grid_end;
    }
    if(this->root == NULL){
        this->learn(X,y);
    }else{
        dVector yprev = this->predict(X);
        this->root = update_tree(X,y,0,yprev);
    } 
}

Node* Method2::update_tree(dMatrix  &X, dVector &y, int depth, dVector &yprev){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    
    if (depth>= this->max_depth){
        return new Node(y.array().mean(),y.rows());
    }
    if(X.rows()<1 || y.rows()<1){
        return NULL;
    }
    if(X.rows()<2 || y.rows()<2){
        return new Node(y.array()(0), y.rows()); 
    }
    if(y.rows()< this->min_split_sample){
        return new Node(y.array().mean() ,y.rows());
    }
    if(all_same(y)){
        return new Node(y.array()(0), y.rows());
    }
    if(all_same_features_values(X)){
        return new Node(y.array().mean() ,y.rows());
    }
    

    bool any_split;
    double score;
    double split_value;
    double impurity;
    int split_feature;
    double w_var;
    iVector mask_left;
    iVector mask_right;
    tie(any_split, split_feature, impurity, score, split_value,w_var)  = find_split_update(X,y, yprev);
    if(!any_split){
        return new Node(y.array().mean() ,y.rows());
    }
    dVector feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, split_value);


    Node* node = new Node(split_value,impurity, score, split_feature, y.rows() , y.array().mean());
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dVector yprev_left = yprev(mask_left,1);
    node->left_child = update_tree( X_left, y_left, depth+1,yprev_left);
   
    dVector yprev_right = yprev(mask_right,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    node->right_child = update_tree(X_right, y_right,depth+1,yprev_right) ;

    return node;
    
}







