#pragma once
#ifndef __TREE_HPP_INCLUDED__

#define __TREE_HPP_INCLUDED__


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
        explicit Tree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity); 
        explicit Tree(); 
        bool all_same(dVector &vec);
        bool all_same_features_values(dMatrix  &X);
        virtual Node* build_tree(dMatrix  &X, dVector &y, int depth);
        tuple<iVector, iVector> get_masks(dVector &feature, double value);
        virtual void learn(dMatrix  &X, dVector &y);
        Node* get_root();
        double predict_obs(dVector  &obs);
        dVector predict(dMatrix  &X);
        virtual void update(dMatrix &X, dVector &y);
        virtual tuple<bool,int,double, double,double,double> find_split(dMatrix &X, dVector &y);
        Node* update_tree_info(dMatrix &X, dVector &y, Node* node, int depth);
        //~Tree();
        std::vector<Node*> make_node_list();
        virtual Tree* copy();

        int tree_depth;
        
    protected:
        Splitter* splitter;
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        dMatrix cir_sim;
        double grid_end;
        
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
}


 Tree* Tree::copy(){
    Tree* tree = new Tree(*this);
    tree->root = root->copy();
    return tree;
 }
 
 /*
Tree::~Tree(){
    delete root;
    max_depth = NULL;
    adaptive_complexity = NULL;
    _criterion = NULL;
    min_split_sample = NULL;
    min_samples_leaf = NULL;
    total_obs = NULL;
    grid_end = NULL;
    tree_depth = NULL;
    number_of_nodes = NULL;
}
*/

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

} 

tuple<bool,int,double, double,double,double>  Tree::find_split(dMatrix &X, dVector &y){
    
    return splitter->find_best_split(X, y);
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
    set_seed(1); //set seed for cir
    if(adaptive_complexity){
        cir_sim = cir_sim_mat(100, 100);
        grid_end = 1.5*cir_sim.maxCoeff();
    }
    
    total_obs = y.size();
    splitter = new Splitter(min_samples_leaf,total_obs,_criterion, adaptive_complexity);
    if(adaptive_complexity){
        splitter->cir_sim = cir_sim;
        splitter->grid_end = grid_end;
    }
    this->root = build_tree(X, y, 0);
    //printf("done building \n");
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
    number_of_nodes +=1;

    tree_depth = max(depth,tree_depth);
    if (depth>= this->max_depth){
        double y_var = 0.0;
        double pred = y.array().mean();
        if(y.rows()>1){
            y_var = (y.array() - pred).square().mean();
        }
        
        return new Node(pred,y.rows(),y_var,0.0 );
    }
    if(X.rows()<1 || y.rows()<1){
        
        return NULL;
    }
    if(X.rows()<2 || y.rows()<2){
       
        return new Node(y.array()(0), y.rows(),0.0,0.0); 
    }
    if(y.rows()< this->min_split_sample){
        double y_var = 0.0;
        double pred = y.array().mean();
        if(y.rows()>1){
            y_var = (y.array() - pred).square().mean();
        }
      
        return new Node(pred, y.rows(),y_var,0.0);
    }
    
    if(all_same(y)){
     
        return new Node(y.array()(0), y.rows(),0.0,0.0);
    }
    if(all_same_features_values(X)){
        double y_var = 0.0;
        double pred = y.array().mean();
        if(y.rows()>1){
            y_var = (y.array() - pred).square().mean();
        }
 
        return new Node(pred, y.rows(),y_var,0.0 );
    }
    if(this->_criterion ==1){
        if(y.rows()==2 && (y.array()(0) ==0.0  || y.array()(1) ==0.0  ) ){
            double y_var = 0.0;
            double pred = y.array().mean();
            if(y.rows()>1){
                y_var = (y.array() - pred).square().mean();
            }
            return new Node(pred, y.rows(),y_var,0.0 );
        }
    }
    
    //printf("y size %d, min_split_sample %d\n", y.size(),this->min_split_sample);

        
    bool any_split;
    double score;
    double impurity;
    double split_value;
    double w_var;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    tie(any_split, split_feature,impurity, score, split_value,w_var)  = find_split(X,y);
    //printf("%d %d %f \n", any_split,split_feature, impurity);
    if(!any_split){
         return new Node(y.array().mean() ,y.rows() );
    }
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
    tie(mask_left, mask_right) = get_masks(feature, split_value);

    double y_var = 0.0;
    double pred = y.array().mean();
    if(y.rows()>1){
        y_var = (y.array() - pred).square().mean();
    }
    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , pred, y_var, w_var);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    node->left_child = build_tree( X_left, y_left, depth+1);
   
    
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    node->right_child = build_tree(X_right, y_right,depth+1) ;

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
    //printf("%d, %d, %d, %f", this->max_depth,this->number_of_nodes,this->tree_depth, this->min_split_sample);
}

Node* Tree::get_root(){
    return this->root;
}

#endif