#pragma once

#include "tree.hpp"
#include "probabalisticsplitter.hpp"

class SemiRandomTree: public Tree{
    public:
        SemiRandomTree(int max_depth, double min_split_sample, int ntrees);
        SemiRandomTree();
        virtual SemiRandomTree* find_best_tree(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval);
    private:
        int ntrees;
        double mean_squared_error(dVector &y_true, dVector &y_pred);
};

SemiRandomTree::SemiRandomTree():Tree(){
    Tree();
    this->splitter = ProbabalisticSplitter();
    this->ntrees = 100;
}

SemiRandomTree::SemiRandomTree(int max_depth, double min_split_sample, int ntrees):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
    this->splitter = ProbabalisticSplitter();
    this->ntrees = ntrees;
}

double SemiRandomTree::mean_squared_error(dVector &y_true, dVector &y_pred){
    return (y_true.array() - y_pred.array()).pow(2.0).mean();
}

SemiRandomTree* SemiRandomTree::find_best_tree(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval){
    SemiRandomTree* bestTree;
    int bestind = 0;
    double bestscore = std::numeric_limits<double>::infinity();
    for (int i = 0; i < this->ntrees; i++)
    {
        SemiRandomTree* t = new SemiRandomTree(this->max_depth, this->min_split_sample, this->ntrees);
        t->learn(X,y);
        dVector y_pred = t->predict(X_eval);
        double score = mean_squared_error(y_eval,y_pred);
        printf("%f, %f \n",bestscore,score);
        if(score<bestscore){
            
            bestTree = t;
            bestscore = score;
            bestind = i;
        }else{
            delete t;
        }
    }
    return bestTree;
}

