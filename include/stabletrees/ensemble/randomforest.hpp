#pragma once
#include "tree.hpp"
#include "naiveupdate.hpp"
#include "treereevaluation.hpp"
#include "stabilityregularization.hpp"
#include "abutreeI.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include <omp.h>
#include <random>


class RandomForest{
    public:
        explicit RandomForest(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,int method );
        void update(dMatrix &X, dVector &y);
        void learn(dMatrix &X, dVector &y);
        dVector predict(dMatrix &X);
        tuple<dMatrix,dVector> RandomForest::sample_X_y(const dMatrix &X,const dVector &y);
        

    protected:
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        int max_features;
        int n_estimator;
        double initial_pred;
        std::vector<Tree*> forest;
        unsigned int random_state;
        int method;
        double lambda;
        double delta;
        double alpha;
        Tree* create_tree(int method,int random_state_);
        iMatrix sample_indices(int start, int end);
        iMatrix bootstrap_indices;

};

RandomForest::RandomForest(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, int method ){
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }
    this-> min_split_sample = min_split_sample;
    this->min_samples_leaf = min_samples_leaf;
    this->_criterion = _criterion;
    this->adaptive_complexity = adaptive_complexity;
    this->max_features = max_features;
    this->n_estimator = n_estimator;
    std::vector<Tree*> forest;
    thread_local unsigned int random_state = 0;
    this->method = method;
    this->lambda = 0.25;
    this->delta = 0.1;
    this->alpha = 0.05;
}

Tree* RandomForest::create_tree(int method, int random_state_){
    switch (method) {
        case 0:
            return new Tree(this->_criterion,this->max_depth, this->min_split_sample, this->min_samples_leaf, this->adaptive_complexity, this->max_features, 1, 0);
        case 1:
            return new NaiveUpdate(this->_criterion,this->max_depth, this->min_split_sample, this->min_samples_leaf, this->adaptive_complexity, this->max_features, 1, 0);
        case 2:
            return new TreeReevaluation(alpha,delta,this->_criterion,this->max_depth, this->min_split_sample, this->min_samples_leaf, this->adaptive_complexity, this->max_features, 1, 0);
        case 3:
            return new StabilityRegularization(lambda,  _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, 0);
        case 4:
            return new AbuTreeI( _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, 0);
        default:
            throw exception("Invalid method");
            break;
    }
}

void RandomForest::learn(dMatrix &X, dVector &y){
    random_state = 0;
    bootstrap_indices = sample_indices(0, y.size());
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    int max_threads = omp_get_max_threads();
    this->forest.clear();
    max_threads = min(max_threads-1,n_estimator );
    for (size_t i = 0; i < n_estimator; i++)
    {   
        Tree* tree = create_tree(method,i);
        this->forest.push_back(tree);     
    }

    //#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < n_estimator; i++)
    {   
        iVector ind = bootstrap_indices.col(i);
        dMatrix X_b = X(ind,keep_cols);
        dVector y_b = y(ind);
        //tie(X_b,y_b) = sample_X_y(X,y);
     
        this->forest[i]->learn(X_b,y_b);
        //printf("learned\n");
    }

}

void RandomForest::update(dMatrix &X, dVector &y){
    random_state = 0;
    int max_threads = omp_get_max_threads();
    iMatrix combined(X.rows(), n_estimator);
    
    //iMatrix bootstrap_indices_new = sample_indices(X.rows()-bootstrap_indices.rows(), X.rows());
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    //combined << bootstrap_indices, bootstrap_indices_new;
    //bootstrap_indices = combined;
    bootstrap_indices =  sample_indices(0, y.size());
    #pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < n_estimator; i++)
    {   
        iVector ind = bootstrap_indices.col(i);
        dMatrix X_b = X(ind,keep_cols);
        dVector y_b = y(ind);
        this->forest[i]->update(X_b,y_b);
        //printf("updated\n");
    }
}

dVector RandomForest::predict(dMatrix &X){
    
    dVector predictions = dVector::Zero(X.rows(),1);
    int max_threads = omp_get_max_threads();
    #pragma omp parallel for num_threads(4)  schedule(dynamic) shared(predictions)
    for (size_t i = 0; i < n_estimator; i++)
    {   
        dVector pred = this->forest[i]->predict(X);
        #pragma omp critical
        predictions = predictions + pred;
    }
    
    
    return predictions.array()/n_estimator;
}
        
        

tuple<dMatrix,dVector> RandomForest::sample_X_y(const dMatrix &X,const dVector &y){
    std::mt19937 gen(random_state);
    size_t n = y.size();
    std::uniform_int_distribution<size_t>  distr(0, n-1);
    dMatrix X_sample(n, X.cols());
    dVector y_sample(n,1);
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = distr(gen);
        for (size_t j = 0; j < X.cols(); j++){
            X_sample(i,j) = X(ind,j);
        }
        y_sample(i,0) = y(ind,0);
    }   
    return tuple<dMatrix,dVector>(X_sample,y_sample);
}


iMatrix RandomForest::sample_indices(int start, int end){
    std::uniform_int_distribution<int>  distr(start, end-1);
    iMatrix bootstrap_indices_(end-start,this->n_estimator);
    int max_threads = omp_get_max_threads();
    #pragma omp parallel for num_threads(max_threads-2)  schedule(dynamic)
    for (int b = 0; b < n_estimator; b++) {
        std::mt19937 gen(b);
        for (int i = 0; i < end-start; i++) {
            bootstrap_indices_(i,b) = distr(gen);;
        }
    }
    return bootstrap_indices_;
}
