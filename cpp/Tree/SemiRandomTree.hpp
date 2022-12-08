#pragma once

#include "tree.hpp"
#include "probabalisticsplitter.hpp"

class SemiRandomTree: public Tree{
    public:
        
        SemiRandomTree(int max_depth, double min_split_sample, int random_state);
        SemiRandomTree();
        virtual void update(dMatrix &X, dVector &y, dVector yprev);
        
        virtual tuple<int, double,double>  SemiRandomTree::find_split(dMatrix &X, dVector &y, dVector &yprev);
        //virtual SemiRandomTree* find_best_tree(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval);
    private:
        Node* update_tree(dMatrix  &X, dVector &y, int depth, dVector &yprev);
        int seed;
};


SemiRandomTree::SemiRandomTree(int max_depth, double min_split_sample, int random_state):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
    this->seed = random_state;
}

tuple<int, double,double>  SemiRandomTree::find_split(dMatrix &X, dVector &y, dVector &yprev){
    this->seed++;
    this->seed = this->seed % 100000;
    return ProbabalisticSplitter(this->seed).find_best_split(X, y, yprev);
}


void SemiRandomTree::update(dMatrix &X, dVector &y,dVector yprev) {
    this->root = update_tree(X,y,0,yprev);
}

Node* SemiRandomTree::update_tree(dMatrix  &X, dVector &y, int depth, dVector &yprev){
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
    tie(split_feature, score, split_value)  = find_split(X,y, yprev);
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



