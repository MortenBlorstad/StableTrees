#pragma once

#include "tree.hpp"
#include "probabalisticsplitter.hpp"

class SemiRandomTree: public Tree{
    public:
        
        SemiRandomTree(int max_depth, double min_split_sample, int random_state);
        SemiRandomTree();
        
        virtual tuple<int, double,double>  SemiRandomTree::find_split(dMatrix &X, dVector &y);
        //virtual SemiRandomTree* find_best_tree(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval);
    private:
        int seed;
};


SemiRandomTree::SemiRandomTree(int max_depth, double min_split_sample, int random_state):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
    this->seed = random_state;
}

tuple<int, double,double>  SemiRandomTree::find_split(dMatrix &X, dVector &y){
    this->seed++;
    this->seed = this->seed % 100000;
    return ProbabalisticSplitter(seed).find_best_split(X, y);
}






