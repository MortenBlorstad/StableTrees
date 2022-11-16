#pragma once
#include "tree.hpp"
#include "stablesplitter.hpp"

class StableTreeReg: public Tree{
    public:
        StableTreeReg(int max_depth, double min_split_sample);
        StableTreeReg();
        virtual void update(dMatrix &X, dVector &y, const string &method = "method1");
        void update_method1(dMatrix &X, dVector &y);
        void update_method2(dMatrix &X, dVector &y);
        void update_method3(dMatrix &X, dVector &y, dVector y_pred1, dVector y_pred2);

    private:
        tuple<Node*, bool> reevaluate_split(Node* node, dMatrix &X, dVector &y, double delta, int depth);
        Node* attempt_split(Node* node, dMatrix &X, dVector &y, int depth);
        double hoeffding_bound(double delta, int n);
        Node* update_tree(dMatrix  &X, dVector &y, int depth, dVector tree1_predictions);
        Node* update_tree(dMatrix  &X, dVector &y, int depth, dVector tree1_predictions, dVector tree2_predictions);
        StableSplitter stable_splitter;

};

StableTreeReg::StableTreeReg():Tree(){
    Tree();
    stable_splitter = StableSplitter();
}

StableTreeReg::StableTreeReg(int max_depth, double min_split_sample):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
    printf("min_split_sample is %f \n", this-> min_split_sample);
}

Node* StableTreeReg::update_tree(dMatrix  &X, dVector &y, int depth, dVector tree1_predictions){
    if (depth> this->max_depth){
        return new Node(y.array().mean(),y.rows());
    }
    if(all_same(y)){
        // printf("all the same \n");
        return new Node(y.array()(0), y.rows());
    }
    if(all_same_features_values(X)){
        return new Node(y.array().mean() ,y.rows());
    }
    if(y.rows()< this->min_split_sample){
        return new Node(y.array().mean() ,y.rows());
    }


    double score;
    double split_value;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    tie(split_feature, score, split_value)  = stable_splitter.find_best_split(X,y,tree1_predictions);
    dVector feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, y, split_value);
    if(mask_left.rows()== X.rows()){
        return new Node(y.array().mean(),y.rows());
    }
    Node* node = new Node(split_value, score, split_feature, y.rows() , y.array().mean());
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dVector tree1_predictions_left = tree1_predictions(mask_left,1);
    
    node->left_child = update_tree( X_left, y_left, depth+1,tree1_predictions_left);
    
    
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector tree1_predictions_right = tree1_predictions(mask_right,1);
    node->right_child = update_tree(X_right, y_right,depth+1,tree1_predictions_right);

    return node;
    
}

void StableTreeReg::update_method1(dMatrix &X, dVector &y){
    if(root == NULL){
        learn(X,y);
    }else{
        dVector tree1_predictions = this->predict(X);
        this->root = update_tree(X,y,0,tree1_predictions);
        
    } 
} 

void StableTreeReg::update(dMatrix &X, dVector &y, const string &method){
    
    if(method == "method2"){
        update_method2(X,y);
    }else{

        update_method1(X,y);
    }
        
        
    
} 

Node* StableTreeReg::update_tree(dMatrix  &X, dVector &y, int depth, dVector tree1_predictions, dVector tree2_predictions){
    if (depth> this->max_depth){
        return new Node(y.array().mean(),y.rows());
    }
    if(all_same(y)){
        // printf("all the same \n");
        return new Node(y.array()(0), y.rows());
    }
    if(all_same_features_values(X)){
        return new Node(y.array().mean() ,y.rows());
    }
    if(y.rows()< this->min_split_sample){
        return new Node(y.array().mean() ,y.rows());
    }


    double score;
    double split_value;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    tie(split_feature, score, split_value)  = stable_splitter.find_best_split(X,y,tree1_predictions,tree2_predictions);
    dVector feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, y, split_value);
    if(mask_left.rows()== X.rows()){
        return new Node(y.array().mean(),y.rows());
    }
    Node* node = new Node(split_value, score, split_feature, y.rows() , y.array().mean());
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dVector tree1_predictions_left = tree1_predictions(mask_left,1);
    dVector tree2_predictions_left = tree2_predictions(mask_left,1);
    
    node->left_child = update_tree( X_left, y_left, depth+1,tree1_predictions_left,tree2_predictions_left);
    
    
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector tree1_predictions_right = tree1_predictions(mask_right,1);
    dVector tree2_predictions_right = tree2_predictions(mask_right,1);
    node->right_child = update_tree(X_right, y_right,depth+1,tree1_predictions_right,tree2_predictions_right);

    return node;
    
}

void StableTreeReg::update_method3(dMatrix &X, dVector &y, dVector y_pred1, dVector y_pred2){
    if(root == NULL){
        learn(X,y);
    }else{
        this->root = update_tree(X,y,0,y_pred1,y_pred2);
    } 
} 

void StableTreeReg::update_method2(dMatrix &X, dVector &y){
    if(root == NULL){
        learn(X,y);
    }else{
        dVector tree1_predictions = this->predict(X);
        Tree* tree2 = new Tree(this->min_split_sample,this->max_depth);
        tree2->learn(X,y);
        dVector tree2_predictions = tree2->predict(X);
        this->root = update_tree(X,y,0,tree1_predictions,tree2_predictions);
    } 
} 



