#pragma once
#ifndef __TREE_HPP_INCLUDED__

#define __TREE_HPP_INCLUDED__

//#include <C:\Users\mb-92\OneDrive\Skrivebord\studie\StableTrees\cpp\thirdparty\eigen\Eigen/Dense>
#include <Eigen/Dense>
#include "node.hpp"
#include "splitter.hpp"


using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;


using namespace std;
using namespace Eigen;

#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort


class Tree{

    public:
        
        Node* root  = NULL;
        explicit Tree(int _criterion,int max_depth, double min_split_sample); 
        explicit Tree(); 
        bool all_same(dVector &vec);
        bool all_same_features_values(dMatrix  &X);
        Node* build_tree(dMatrix  &X, dVector &y, int depth);
        tuple<iVector, iVector> get_masks(dVector &feature, dVector &y, double value);
        virtual void learn(dMatrix  &X, dVector &y);
        Node* get_root();
        double predict_obs(dVector  &obs);
        dVector predict(dMatrix  &X);
        virtual void update(dMatrix &X, dVector &y);
        virtual tuple<int,double, double,double> find_split(dMatrix &X, dVector &y);
    protected:
        //Splitter splitter;
        int max_depth;
        int _criterion;
        double min_split_sample;
};
Tree::Tree(){
    int max_depth = INT_MAX;
    double min_split_sample = 2.0;
    _criterion = 0;

}

Tree::Tree(int _criterion, int max_depth, double min_split_sample){
    //this->splitter = Splitter(_criterion);
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }

    this-> min_split_sample = min_split_sample;
    this->_criterion = _criterion;
} 

tuple<int,double, double,double>  Tree::find_split(dMatrix &X, dVector &y){
    return Splitter(this->_criterion).find_best_split(X, y);
}


bool Tree::all_same(dVector &vec){
    bool same = true;
    for(int i=0; i< vec.rows(); i++){
        if(vec(i)!=vec(0) ){
            same=false;
            break;
        }
    }
    return same;
}

bool Tree::all_same_features_values(dMatrix &X){
    bool same = true;
    dVector feature;
    for(int i =0; i<X.cols(); i++){
        feature = X.col(i);
        if(!all_same(feature)){
            same=false;
            break;
        }
    }
    return same;
}

tuple<iVector, iVector> Tree::get_masks(dVector &feature, dVector &y, double value){
    std::vector<int> left_values;
    std::vector<int> right_values;
    for(int i=0; i<y.rows();i++){
        if(feature[i]<=value){
            left_values.push_back(i);
        }else{
            right_values.push_back(i);
        }
    }
    iVector left_values_v = Eigen::Map<iVector, Eigen::Unaligned>(left_values.data(), left_values.size());
    iVector right_values_v = Eigen::Map<iVector, Eigen::Unaligned>(right_values.data(), right_values.size());

    return tuple<iVector, iVector> (left_values_v, right_values_v);
}



void Tree::learn(dMatrix  &X, dVector &y){
    this->root = build_tree(X, y, 0);
}

double Tree::predict_obs(dVector  &obs){
    Node* node = this->root;
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            return node->predict();
        }else{
            //printf("feature %d, value %f, obs %f \n", node->split_feature, node->split_value,obs(node->split_feature));
            if(obs(node->split_feature) <= node->split_value){
                node = node->left_child;
            }else{
                node = node->right_child;
            }
        }
    }
    return NULL;
}

dVector Tree::predict(dMatrix  &X){
    int n = X.rows();
    dVector y_pred(n);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        y_pred[i] = predict_obs(obs);
    }
    return y_pred;

}



Node* Tree::build_tree(dMatrix  &X, dVector &y, int depth){
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
        return new Node(y.array().mean(), y.rows());
    }
    if(all_same(y)){
        return new Node(y.array()(0), y.rows());
    }
    if(all_same_features_values(X)){
        return new Node(y.array().mean() ,y.rows());
    }
    if(this->_criterion ==1){
        if(y.rows()==2 && (y.array()(0) ==0.0  || y.array()(1) ==0.0  ) ){
            return new Node(y.array().mean() ,y.rows());
        }
    }
    
    //printf("y size %d, min_split_sample %d\n", y.size(),this->min_split_sample);

        

    double score;
    double impurity;
    double split_value;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    tie(split_feature,impurity, score, split_value)  = find_split(X,y);
    //printf("score %d\n", score);
    //printf("score %f\n", score);
    if(score == std::numeric_limits<double>::infinity()){
        cout<<"\n Two Dimensional Array is : \n";
        for(int r=0; r<X.rows(); r++)
        {
                for(int c=0; c<X.cols(); c++)
                {
                        cout<<" "<<X(r,c)<<" ";
                }
                cout<<"\n";
        }
         cout<<"\n one Dimensional Array is : \n";
        
        for(int c=0; c<y.size(); c++)
        {
                cout<<" "<<y(c)<<" ";
        }
        cout<<"\n";
        
    }
   

    dVector feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, y, split_value);


    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , y.array().mean());
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    node->left_child = build_tree( X_left, y_left, depth+1);
   
    
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    node->right_child = build_tree(X_right, y_right,depth+1) ;

    return node;
    

}

void Tree::update(dMatrix &X, dVector &y){
    this->learn(X,y);
    printf("%d, %f", this->max_depth, this->min_split_sample);
}

Node* Tree::get_root(){
    return this->root;
}

#endif