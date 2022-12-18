#pragma once
#include "tree.hpp"
#include "splitterReg.hpp"
class Method2: public Tree{
    public:
        Method2(int _criterion,int max_depth, double min_split_sample);
        Method2();
        virtual void update(dMatrix &X, dVector &y);
        

    private:
        Node* update_tree(dMatrix  &X, dVector &y, int depth, dVector &yprev);
};

Method2::Method2():Tree(){
    Tree();
}

Method2::Method2(int _criterion,int max_depth, double min_split_sample):Tree(_criterion, max_depth,  min_split_sample){
    Tree(_criterion, max_depth, min_split_sample);
}
   
    
void Method2::update(dMatrix &X, dVector &y) {
    if(this->root == NULL){
        this->learn(X,y);
    }else{
        dVector yprev = this->predict(X);
        this->root = update_tree(X,y,0,yprev);
    } 
}

Node* Method2::update_tree(dMatrix  &X, dVector &y, int depth, dVector &yprev){
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


    double score;
    double split_value;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    tie(split_feature, score, split_value)  = SplitterReg(this->_criterion).find_best_split(X,y, yprev);
    dVector feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, y, split_value);


    Node* node = new Node(split_value, score, split_feature, y.rows() , y.array().mean());
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dVector yprev_left = yprev(mask_left,1);
    node->left_child = update_tree( X_left, y_left, depth+1,yprev_left);
   
    dVector yprev_right = yprev(mask_right,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    node->right_child = update_tree(X_right, y_right,depth+1,yprev_right) ;

    return node;
    
}





