#pragma once
#include "tree.hpp"
#include "splitter.hpp"

#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

using namespace std;

class Method1: public Tree{
    public:
        Method1(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity);
        Method1();
        virtual void update(dMatrix &X, dVector &y, double delta);
        std::vector<double> get_eps();
        std::vector<double> get_mse_ratio();
        std::vector<double> get_obs();
        std::vector<double>  mse_ratio; 
        std::vector<double> epss; 
        std::vector<double> obs; 
    private:
        tuple<Node*, bool> reevaluate_split(Node* node, dMatrix &X, dVector &y, double delta, int depth);
        Node* attempt_split(Node* node, dMatrix &X, dVector &y, int depth);
        Node* update_rec(Node* node, dMatrix &X, dVector &y,double delta, int depth);
        double hoeffding_bound(double delta, int n);
        int number_of_examples;
        vector<size_t> sort_index(const vector<double> &v);
        

};

std::vector<double> Method1::get_mse_ratio(){
    return mse_ratio;
}
std::vector<double> Method1::get_eps(){
    return epss;
}
std::vector<double> Method1::get_obs(){
    return obs;
}


Method1::Method1():Tree(){
    Tree();
}

Method1::Method1(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity);
}

vector<size_t> Method1::sort_index(const vector<double> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

void Method1::update(dMatrix &X, dVector &y, double delta){
    if(root == NULL){
        learn(X,y);
    }

    number_of_examples = y.size();
    root = update_rec(root, X, y, delta,0);
    root = update_tree_info(X, y, root,0);

    vector<size_t> sorted_ind = sort_index(obs);

    std::vector<double>  mse_ratio_sorted; 
    std::vector<double> epss_sorted;
    std::vector<double> obs_sorted;
    for (auto i: sorted_ind) {
        mse_ratio_sorted.push_back(mse_ratio[i]);
        epss_sorted.push_back(epss[i]);
        obs_sorted.push_back(obs[i]); 
    }
   
    mse_ratio = mse_ratio_sorted;
    epss = epss_sorted;
    obs = obs_sorted;


} 


Node* Method1::update_rec(Node* node, dMatrix &X, dVector &y,double delta,int depth){
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
        tie(left_mask,right_mask) = get_masks(feature, node->split_value);
        iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
        
        dMatrix X_left = X(left_mask,keep_cols); dVector y_left = y(left_mask,1);
        node->left_child = update_rec(node->left_child, X_left, y_left,delta,depth+1);
        dMatrix X_right= X(right_mask,keep_cols); dVector y_right = y(right_mask,1);
        node->right_child = update_rec(node->right_child, X_right, y_right,delta,depth+1);
    }
    return node;
}

double Method1::hoeffding_bound(double delta, int n){
    return sqrt(log(1/delta)/(2*n));
}

Node* Method1::attempt_split(Node* node, dMatrix &X, dVector &y, int depth){
    if(max_depth<= depth || y.rows() <2 || y.rows()<min_split_sample){
        node->n_samples = y.rows();
        return node;
    }
    return build_tree(X,y,depth);
}



tuple<Node*, bool> Method1::reevaluate_split(Node* node, dMatrix &X, dVector &y, double delta, int depth){
    bool any_split;
    double new_score;
    double w_var;
    double new_impurity;
    double split_value;
    int split_feature;
    double old_score = node->get_split_score();
    bool changed = false;
    
    tie(any_split,split_feature,new_impurity, new_score, split_value,w_var) = find_split(X,y);
    node->n_samples = y.rows();
    double eps = this->hoeffding_bound(delta, node->n_samples);
    if((old_score+1)/(new_score+1)> (eps+1.05) && any_split){
        node = build_tree(X,y,depth);
        changed = true;
    }
    mse_ratio.push_back((old_score+1)/(new_score+1));
    epss.push_back(eps+1);
    obs.push_back((double)node->n_samples);
    
    return  tuple<Node*, bool>(node, changed);
}

