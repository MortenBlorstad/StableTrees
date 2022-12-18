#pragma once
#ifndef __Splitter_HPP_INCLUDED__
#include "splitter.hpp"
#include "node.hpp"
#include <stdexcept>

class SplitterReg: public Splitter{
    public:
        SplitterReg();
        SplitterReg(int _citerion);
        tuple<int,double,double>  find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev);
        tuple<double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index,const dVector &y_prev);
    protected:
        Criterion *criterion;
};

SplitterReg::SplitterReg(){
    MSEReg crit;
    criterion = &crit;
}

SplitterReg::SplitterReg(int _citerion){
    if(_citerion == 0){
        MSEReg crit;
        criterion = &crit;
    }else if(_citerion ==1){
        PoissonReg crit;
        criterion = &crit;
    }else{
        throw invalid_argument("Possible criterions are 'mse' and 'poisson'.");
    }
}


tuple<double,double> SplitterReg::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index,const dVector &y_prev){
    //https://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset
    double min_score = std::numeric_limits<double>::infinity();
    double n = y.size();
    double best_split_value;
    
    double largestValue = feature(sorted_index[n-1]);


    double yp_L = 0; // sum of predictions from tree 1, left
    double yp_R = y_prev.array().sum();  // sum of predictions from tree 1, right
    double yp_squared = y_prev.array().square().sum();  // sum of squares of predictions from tree


    for (int i = 0; i < n-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];
        
        double split_value =  (lowValue+hightValue)/2;
        criterion->update(y[low],y_prev(low));
        
    
        // break if rest of the values are equal
        if(lowValue == largestValue){
            break;
        }
        // skip if values are approx equal
        if(hightValue-lowValue<0.000001){
            continue;
        }
        if(criterion->should_skip()){
            continue;
        }
        double score = criterion->get_score();

        
        if(min_score>score){
            min_score = score;
            best_split_value = split_value;
        }
    }
  
    
    return tuple<double,double>(min_score,best_split_value);
}


tuple<int, double,double> SplitterReg::find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev){
        
        double min_score = std::numeric_limits<double>::infinity();
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        double split_value;
        int i;
        int n = y.size();
        criterion->init(n,y,y_prev);
        

        iMatrix X_sorted_indices = sorted_indices(X);
    
        //#pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature) private(i,score,split_value, feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
            tie(score, split_value) = select_split_from_all(feature, y, sorted_index,y_prev);
            
        //   #pragma omp ordered
        //{
            if(feature[sorted_index[0]] != feature[sorted_index[feature.rows()-1]]){
                if(min_score>score){
                    min_score = score;
                    best_split_value = split_value;
                    split_feature = i;
                }
            }
           // }
           criterion->reset();
        }
        return tuple<int, double,double>(split_feature,min_score, best_split_value);
    
    
}
#endif