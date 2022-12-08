#pragma once

#include "tree.hpp"
#include "SemiRandomTree.hpp"
#include "probabalisticsplitter.hpp"

class MonteCarloTree: public Tree{
    public:
        MonteCarloTree(int max_depth, double min_split_sample, int ntrees, int random_state);
        MonteCarloTree();
        virtual void update(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval);
    private:
        int seed;
        int ntrees;
        double mean_squared_error(dVector &y_true, dVector &y_pred);
        double stability(dVector &y_true, dVector &y_pred, dVector &y_pred1);
};

MonteCarloTree::MonteCarloTree():Tree(){
    Tree();
}

   
MonteCarloTree::MonteCarloTree(int max_depth, double min_split_sample, int ntrees, int random_state):Tree( max_depth,  min_split_sample){
    Tree(max_depth, min_split_sample);
    this->ntrees = ntrees;
    this->seed = random_state;
}

double MonteCarloTree::mean_squared_error(dVector &y_true, dVector &y_pred){
    return (y_true.array() - y_pred.array()).pow(2.0).mean();
}

double MonteCarloTree::stability(dVector &y_true, dVector &y_pred, dVector &y_pred1){
    //return mean_squared_error(y_true,y_pred) ;//+ (y_pred1.array() - y_pred.array()).pow(2.0).mean();
    return (y_pred1.array() - y_pred.array()).abs().mean();
}


void MonteCarloTree::update(dMatrix &X, dVector &y, dMatrix &X_eval, dVector &y_eval){
    double bestscore = std::numeric_limits<double>::infinity();
    dVector ypred1 = predict(X_eval);
    
    SemiRandomTree* bestTree = new SemiRandomTree(this->max_depth, this->min_split_sample, this->seed);
    //printf("%d \n", ntrees);
    #pragma omp parallel for ordered num_threads(4) shared(bestTree, bestscore)
    for (int i = 0; i < this->ntrees; i++)
    {   
        //printf("%d \n", i);
        SemiRandomTree* t = new SemiRandomTree(this->max_depth, this->min_split_sample, this->seed + i);

        t->update(X,y,ypred1);

        dVector y_pred = t->predict(X_eval);

        double score = stability(y_eval,y_pred,ypred1);

       #pragma omp ordered
        {
        if(score<bestscore){
            //printf("%d, %f, %f, %d \n ", this->seed + i, score,bestscore,y.size());
            bestTree = t;
            bestscore = score;
        }else{
  
            delete t;
        }
        }
    }
    this->root = bestTree->root;
}

