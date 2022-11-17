#pragma once

#include "tree.hpp"
#include "SemiRandomTree.hpp"
#include "probabalisticsplitter.hpp"

class MonteCarloTree: public Tree{
    public:
        MonteCarloTree(int max_depth, double min_split_sample, int ntrees, int random_state);
        MonteCarloTree();
        virtual void learn(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval);
    private:
        int seed;
        int ntrees;
        double mean_squared_error(dVector &y_true, dVector &y_pred);
};


MonteCarloTree::MonteCarloTree(int max_depth, double min_split_sample, int ntrees, int random_state):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
    this->ntrees = ntrees;
    this->seed = random_state;
}

double MonteCarloTree::mean_squared_error(dVector &y_true, dVector &y_pred){
    return (y_true.array() - y_pred.array()).pow(2.0).mean();
}

void MonteCarloTree::learn(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval){
    SemiRandomTree* bestTree;
    double bestscore = std::numeric_limits<double>::infinity();
    #pragma omp parallel for ordered num_threads(4) shared(bestTree,bestscore)
    for (int i = 0; i < this->ntrees; i++)
    {
        SemiRandomTree* t = new SemiRandomTree(this->max_depth, this->min_split_sample, this->seed + i);
        t->learn(X,y);
        dVector y_pred = t->predict(X_eval);
        double score = mean_squared_error(y_eval,y_pred);
        #pragma omp ordered
        {
        if(score<bestscore){
            bestTree = t;
            bestscore = score;
        }else{
            delete t;
        }
        }
    }
    this->root = bestTree->root;
}

