#pragma once
#include "tree.hpp"


class StableTree: public Tree{
    public:
        StableTree(int max_depth, double min_split_sample);
        StableTree();
        virtual void update(dMatrix &X, dVector &y, double delta);
    private:
        tuple<Node*, bool> reevaluate_split(Node* node, dMatrix &X, dVector &y, double delta, int depth);
        Node* attempt_split(Node* node, dMatrix &X, dVector &y, int depth);
        Node* update_rec(Node* node, dMatrix &X, dVector &y,double delta, int depth);
        double hoeffding_bound(double delta, int n);
        int number_of_examples;

};

StableTree::StableTree():Tree(){
    Tree();
}

StableTree::StableTree(int max_depth, double min_split_sample):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
}



void StableTree::update(dMatrix &X, dVector &y, double delta){
    if(root == NULL){
        learn(X,y);
    }
    number_of_examples = y.size();
    root = update_rec(root, X, y, delta,0);
} 

Node* StableTree::update_rec(Node* node, dMatrix &X, dVector &y,double delta,int depth){
    if(y.size()<=1){
        return node;
    }
    if(node->is_leaf()){
        return attempt_split(node, X,y, depth);
    }

    bool change;
    tie(node,change) = reevaluate_split(node,X,y,delta,depth);
    //printf("change %d, depth: %d \n", change, depth);
    if(!change){
        iVector left_mask; iVector right_mask;
        dVector feature = X.col(node->split_feature);
        tie(left_mask,right_mask) = get_masks(feature,y, node->split_value);
        iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
        
        dMatrix X_left = X(left_mask,keep_cols); dVector y_left = y(left_mask,1);
        node->left_child = update_rec(node->left_child, X_left, y_left,delta,depth+1);
        dMatrix X_right= X(right_mask,keep_cols); dVector y_right = y(right_mask,1);
        node->right_child = update_rec(node->right_child, X_right, y_right,delta,depth+1);
        
    }
    return node;

}

double StableTree::hoeffding_bound(double delta, int n){
    return sqrt(log(1/delta)/(2*n));
}


Node* StableTree::attempt_split(Node* node, dMatrix &X, dVector &y, int depth){
    if(y.rows() <2 || y.rows()<min_split_sample){
        node->n_samples = y.rows();
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
    node->n_samples = y.rows();
    double eps = this->hoeffding_bound(delta, (number_of_examples)/(node->n_samples));
    if((mse_old+1)/(mse_new+1)> (1+eps)){
        node = build_tree(X,y,depth);
        changed = true;
    }else{
        node->split_score = mse_new;
    }

    return  tuple<Node*, bool>(node, changed);
}

