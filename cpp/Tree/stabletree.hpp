#pragma once
#include "tree.hpp"


class StableTree: public Tree{
    public:
        StableTree();
        virtual void update(dMatrix &X, dVector &y);
    private:
        tuple<Node*, bool> reevaluate_split(Node* node, dMatrix &X, dVector &y, double delta, int depth);
        Node* attempt_split(Node* node, dMatrix &X, dVector &y, int depth);
        Node* update_rec(Node* node, dMatrix &X, dVector &y,double delta, int depth);
        double delta;

    
};


StableTree::StableTree(){
    Tree();
    this->delta = 0.1;
}

void StableTree::update(dMatrix &X, dVector &y){
    if(root == NULL){
        learn(X,y);
    }
    root = update_rec(root, X, y, this->delta,0);
} 

Node* StableTree::update_rec(Node* node, dMatrix &X, dVector &y,double delta,int depth){
    if(X.size()<=0){
        return node;
    }
    if(node->is_leaf()){
        return attempt_split(node, X,y, depth);
    }

    bool change;
    tie(node,change) = reevaluate_split(node,X,y,delta,depth);
    if(!change){
        iVector left_mask; iVector right_mask;
        dVector feature = X.col(node->split_feature);
        tie(left_mask,right_mask) = get_masks(feature,y, node->split_value);
        iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()).array();
        if(left_mask.rows()){
            dMatrix X_left = X(left_mask,keep_cols); dVector y_left = y(left_mask,1);
            node->left_child = update_rec(node->left_child, X_left, y_left,delta,depth+1);
        }
        if(right_mask.rows()){
            dMatrix X_right= X(right_mask,keep_cols); dVector y_right = y(right_mask,1);
            node->right_child = update_rec(node->right_child, X_right, y_right,delta,depth+1);
        }
    }
    return node;

}


Node* StableTree::attempt_split(Node* node, dMatrix &X, dVector &y, int depth){
    if(y.rows() <2 || y.rows()<min_split_sample){
        return node;
    }
    return build_tree(X,y,depth);
}

tuple<Node*, bool> StableTree::reevaluate_split(Node* node, dMatrix &X, dVector &y, double delta, int depth){
    
    double mse_new;
    double split_value;
    int split_feature;
    double mse_old = node->get_split_score();
    bool changed = false;
    tie(split_feature, mse_new, split_value) = splitter.find_best_split(X,y);
    if((mse_old+10e-4)/(mse_new+10e-4)> (1-delta)){
        node = build_tree(X,y,depth);
        changed = true;
    }

    return  tuple<Node*, bool>(node, changed);
}

