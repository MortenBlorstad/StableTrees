#pragma once
#ifndef __TREE_HPP_INCLUDED__

#define __TREE_HPP_INCLUDED__


#include <Eigen/Dense>

#include "node.hpp"
#include "splitter.hpp"
#include "lossfunctions.hpp"


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
        explicit Tree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity); 
        explicit Tree(); 
        bool all_same(const dVector &vec);
        bool all_same_features_values(const dMatrix  &X);
        virtual Node* build_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth);
        tuple<iVector, iVector> get_masks(dVector &feature, double value);
        virtual void learn(dMatrix  &X, dVector &y);
        Node* get_root();
        double predict_obs(dVector  &obs);
        dVector predict(dMatrix  &X);
        virtual void update(dMatrix &X, dVector &y);
        virtual tuple<bool,int,double, double,double,double,double,double> find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h);
        Node* update_tree_info(dMatrix &X, dVector &y, Node* node, int depth);
        //~Tree();
        std::vector<Node*> make_node_list();
        virtual Tree* copy();

        int tree_depth;
        
    protected:
        Splitter* splitter;
        LossFunction* loss_function;
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        
        int n1;
        
        int number_of_nodes;
        void make_node_list_rec(Node* node, std::vector<Node*> &l, size_t index );
};
Tree::Tree(){
    int max_depth = INT_MAX;
    double min_split_sample = 2.0;
    _criterion = 0;
    adaptive_complexity = false;
    this->min_samples_leaf = 1;
    tree_depth = 0;
    number_of_nodes = 0;
    loss_function = new LossFunction(0);
}


 Tree* Tree::copy(){
    Tree* tree = new Tree(*this);
    tree->root = root->copy();
    return tree;
 }
 

Tree::Tree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity){
    //this->splitter = Splitter(_criterion);
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }
    this-> min_split_sample = min_split_sample;
    this->min_samples_leaf = min_samples_leaf;
    this->_criterion = _criterion;
    this->adaptive_complexity = adaptive_complexity;
    tree_depth = 0;
    number_of_nodes = 0;
    loss_function = new LossFunction(_criterion);

} 

tuple<bool,int,double, double,double,double,double,double>  Tree::find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h){
    return splitter->find_best_split(X, y, g, h);
}


bool Tree::all_same(const dVector &vec){
    bool same = true;
    for(int i=0; i< vec.rows(); i++){
        if(vec(i)!=vec(0) ){
            same=false;
            break;
        }
    }
    return same;
}

bool Tree::all_same_features_values(const dMatrix &X){
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

tuple<iVector, iVector> Tree::get_masks(dVector &feature, double value){
    std::vector<int> left_values;
    std::vector<int> right_values;
    for(int i=0; i<feature.rows();i++){
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
    total_obs = y.size();
    //printf("min_samples_leaf: %d \n", min_samples_leaf);
    splitter = new Splitter(min_samples_leaf,total_obs,_criterion, adaptive_complexity);
    n1 = total_obs;
    dVector g = loss_function->dloss(y,  dVector::Zero(n1,1));
    dVector h = loss_function->ddloss(y, dVector::Zero(n1,1)); 
    
    this->root = build_tree(X, y,g, h, 0);
    
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
    return y_pred; //loss_function->inverse_link_function(y_pred);
}



Node* Tree::build_tree(const dMatrix  &X, const dVector &y,const dVector &g, const dVector &h, int depth){
    number_of_nodes +=1;

    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        
        return NULL;
    }

   

    
    int n = y.size();
    double pred = y.array().mean();


    if(all_same(y)){
        return new Node(pred, n,1,1);
    }
    
    bool any_split;
    double score;
    double impurity;
    double split_value;
    double w_var;
    double y_var;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    tie(any_split, split_feature, split_value,impurity, score, y_var ,w_var,expected_max_S)  = find_split(X,y, g,h);
    
    
    if(depth>=this->max_depth){
        return new Node(pred, n, y_var,w_var);
    }
    if(y.rows()< this->min_split_sample){
        return new Node(pred, n, y_var,w_var);
    }
    if(!any_split){
         return new Node(pred ,n, y_var, w_var  );
    }

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

    tie(mask_left, mask_right) = get_masks(feature, split_value);

    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , pred, y_var, w_var);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
    dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);
    
    node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1);
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
    }

    node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1) ;
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
    }

    return node;
}

Node* Tree::update_tree_info(dMatrix &X, dVector &y, Node* node, int depth){
    tree_depth = max(tree_depth, depth);
    node->n_samples = y.size();
    if(node->n_samples<1){
        return NULL;
    }else if(node->n_samples<1){
        node->prediction = y[0];
    }else{
        node->prediction = y.array().mean();
    }
    if(node->is_leaf()){
        return node;
    }
    dVector feature = X.col(node->split_feature);
    iVector mask_left;
    iVector mask_right;
    tie(mask_left, mask_right) = get_masks(feature, node->split_value);
    if(mask_left.size()<1 || mask_right.size()<1 ){
        //printf("null %d %d \n", mask_left.size()<1, mask_right.size()<1);
        node->left_child = NULL;
        node->right_child = NULL;
    }else{
        iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
        dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
        dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);

        //printf("update rec %d %d \n", mask_left.size()<1, mask_right.size()<1);

        node->left_child = update_tree_info(X_left, y_left, node->left_child, depth+1) ;
        node->right_child = update_tree_info(X_right, y_right, node->right_child, depth+1) ;
    }
    //printf("update %d \n", node->get_split_feature());
    return node;
}

void Tree::make_node_list_rec(Node* node, std::vector<Node*> &l, size_t index ){
    if(node->is_leaf()){
        return;
    }
    size_t left_index = index*2+1;
    size_t right_index = index*2+2;
    l[left_index] = node->left_child;
    l[right_index] = node->right_child;
    make_node_list_rec(node->left_child, l, left_index);
    make_node_list_rec(node->right_child, l, right_index);

}

std::vector<Node*> Tree::make_node_list(){
    size_t index = 0;
    int max_number_of_nodes = pow(2,tree_depth+1)-1;
    std::vector<Node*> l(max_number_of_nodes);
    Node* node = root;
    l[index] = node;
    make_node_list_rec(node,l, index);

    return l;
}



void Tree::update(dMatrix &X, dVector &y){
    this->learn(X,y);
}

Node* Tree::get_root(){
    return this->root;
}

#endif