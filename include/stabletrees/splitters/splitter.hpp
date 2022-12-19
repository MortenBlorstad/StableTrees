#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#define __SLITTER_HPP_INCLUDED__

//#include <C:\Users\mb-92\OneDrive\Skrivebord\studie\StableTrees\cpp\thirdparty\eigen\Eigen/Dense>
#include <Eigen/Dense>
#include <unordered_set>
#include "MSE.hpp"
#include "criterion.hpp"
#include "poisson.hpp"

#include <omp.h>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;


using namespace std;



class Splitter{

    public:
        Splitter();
        Splitter(int _citerion);
        tuple<int,double,double,double>  find_best_split(const dMatrix  &X, const dVector  &y);
        tuple<double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index);
    
    protected:
        iMatrix sorted_indices(dMatrix X);
        vector<int> sort_index(const dVector &v);
        Criterion *criterion;
};


Splitter::Splitter(){
    MSE mse;
    criterion = &mse;
}

Splitter::Splitter(int _citerion){
    if(_citerion == 0){
        MSE crit;
        criterion = &crit;
    }else if(_citerion ==1){
        Poisson crit;
        criterion = &crit;
    }else{
        throw invalid_argument("Possible criterions are 'mse' and 'poisson'.");
    }
}

tuple<double,double> Splitter::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index){
    //https://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset
    double min_score = std::numeric_limits<double>::infinity();
    double n = y.size();
    double best_split_value;
    

    double largestValue = feature(sorted_index[n-1]);

    for (int i = 0; i < n-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];
        
        double split_value =  (lowValue+hightValue)/2;
        //printf("%f \n", y[low]);
        criterion->update(y[low]);
        //printf("%f \n", criterion.get_score());


        if(lowValue == largestValue){
            break;
        }
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


tuple<int, double, double,double> Splitter::find_best_split(const dMatrix  &X, const dVector  &y){
        
    

        double min_score = std::numeric_limits<double>::infinity();
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        double split_value;
        int i;
        int n = y.size();
        criterion->init(n,y);
        double impurity = criterion->node_score;

        iMatrix X_sorted_indices = sorted_indices(X);
        


        //#pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature) private(i,score,split_value, feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
          
            tie(score, split_value) = select_split_from_all(feature, y, sorted_index);
            //printf("num obs %d,split_value %f , min_score %f \n", feature.size(),split_value,min_score);
            
           //#pragma omp ordered
        //{
            if(feature[sorted_index[0]] != feature[sorted_index[n-1]]){
               //printf("%d, %f \n",i,score);
                if(min_score>score){
                    min_score = score;
                    best_split_value = split_value;
                    split_feature = i;
                }
            }
            //}
            criterion->reset();
        }
        //printf("=== %d, %f, %f  \n ",split_feature,min_score, best_split_value);
        return tuple<int,double, double,double>(split_feature,impurity,min_score, best_split_value);
    
    
}

vector<int> Splitter::sort_index(const dVector &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] <= v[i2];});

  return idx;
}

iMatrix Splitter::sorted_indices(dMatrix X){
    const int nrows = X.rows();
    const int ncols = X.cols();
    iMatrix X_sorted_indices(nrows,ncols);
    
    for(int i = 0; i<ncols; i++){
        vector<int> sorted_ind = sort_index(X.col(i));
        for(int j = 0; j<nrows; j++){
            X_sorted_indices(j,i) = sorted_ind[j];
        }
    }
    return X_sorted_indices;
}


#endif