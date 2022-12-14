#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#define __SLITTER_HPP_INCLUDED__

//#include <C:\Users\mb-92\OneDrive\Skrivebord\studie\StableTrees\cpp\thirdparty\eigen\Eigen/Dense>
#include <Eigen/Dense>
#include <unordered_set>
#include "MSE.hpp"
#include "criterion.hpp"
#include "poisson.hpp"
#include "cir.hpp"
#include "gumbel.hpp"

#include <omp.h>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dArray = Eigen::Array<double,Eigen::Dynamic,1>;

using namespace std;



class Splitter{

    public:
        Splitter();
        Splitter(int min_samples_leaf, double _total_obs, bool _adaptive_complexity);
        Splitter(int min_samples_leaf,double _total_obs, int _citerion, bool _adaptive_complexity);
        tuple<bool,int,double,double,double>  find_best_split(const dMatrix  &X, const dVector  &y);
        tuple<bool,double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index);
        dMatrix cir_sim;
        double grid_end; 
    protected:
        iMatrix sorted_indices(dMatrix X);
        vector<int> sort_index(const dVector &v);
        Criterion * criterion;
        double total_obs;
        int grid_size = 101;
        dVector grid;
        dArray gum_cdf_mmcir_grid;
        bool adaptive_complexity;
        int min_samples_leaf;

};

Splitter::Splitter(){
    MSE mse;
    criterion = &mse;
    adaptive_complexity = false;
    this->min_samples_leaf = 1;
}

Splitter::Splitter(int min_samples_leaf,double _total_obs, bool _adaptive_complexity){

    MSE mse;
    criterion = &mse;
    total_obs = _total_obs;
    adaptive_complexity =_adaptive_complexity;
    this->min_samples_leaf = min_samples_leaf;
}

Splitter::Splitter(int min_samples_leaf, double _total_obs, int _citerion, bool _adaptive_complexity){
    if(_citerion == 0){
        MSE crit;
        criterion = &crit;
    }else if(_citerion ==1){
        Poisson crit;
        criterion = &crit;
    }else{
        throw invalid_argument("Possible criterions are 'mse' and 'poisson'.");
    }
    total_obs = _total_obs;
    adaptive_complexity =_adaptive_complexity;
    this->min_samples_leaf = min_samples_leaf;
}

tuple<bool,double,double> Splitter::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index){
    //https://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset
    double min_score = std::numeric_limits<double>::infinity();
    double n = y.size();
    double best_split_value;
    
    int num_splits = 0;
    double largestValue = feature(sorted_index[n-1]);
    dVector u_store((int)n);
    if(adaptive_complexity){
        u_store = dVector::Zero(n);
    }
    double prob_delta = 1.0/n;
    
    dArray gum_cdf_grid(grid_size);

    bool any_split = false;
    for (int i = 0; i < n-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];
        
        double split_value =  (lowValue+hightValue)/2;
        criterion->update(y[low]);


        if(lowValue == largestValue){
            break;
        }
        if(hightValue-lowValue<0.000001){
            continue;
        }
        
        if(criterion->should_skip(min_samples_leaf)){
            continue;
        }
        if(adaptive_complexity){
            u_store[num_splits] = (i+1)*prob_delta;
        }
        num_splits +=1;
        any_split = true;
        double score = criterion->get_score();
        if(min_score>score){
            min_score = score;
            best_split_value = split_value;
            criterion->observed_reduction = criterion->node_score - score;
        }

    }
    
    if(num_splits<=0){
        any_split = false;
    }else if(adaptive_complexity){
        dVector u = u_store.head(num_splits);
        dArray max_cir = rmax_cir(u, cir_sim); // Input cir_sim!
        if(num_splits>1){
            // Asymptotically Gumbel
                
            // Estimate Gumbel parameters
            dVector par_gumbel = par_gumbel_estimates(max_cir);
            // Estimate cdf of max cir for feature j
            for(int k=0; k< grid_size; k++){ 
                gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
            }

        }else{
            

            // Asymptotically Gumbel
                
            // Estimate Gumbel parameters
            dVector par_gumbel = par_gumbel_estimates(max_cir);
            // Estimate cdf of max cir for feature j
            for(int k=0; k< grid_size; k++){ 
                gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
            }
        }
        // Update empirical cdf for max max cir
            gum_cdf_mmcir_grid *= gum_cdf_grid; 
    }
    

    return tuple<bool,double,double>(any_split, min_score,best_split_value);
}


tuple<bool, int, double, double,double> Splitter::find_best_split(const dMatrix  &X, const dVector  &y){
        
        
        double min_score = std::numeric_limits<double>::infinity();
        
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        bool any_split;
        double split_value;
        int i;
        int n = y.size();
        
        if(adaptive_complexity){
            grid = dVector::LinSpaced( grid_size, 0.0, grid_end );
            gum_cdf_mmcir_grid = dArray::Ones(grid_size);
        }

        criterion->init(n,y);
    

        double impurity = criterion->node_score;

     
        iMatrix X_sorted_indices = sorted_indices(X);
   
        


        //#pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature) private(i,score,split_value, feature)
       
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
          
            tie(any_split, score, split_value) = select_split_from_all(feature, y, sorted_index);
            
            
           //#pragma omp ordered
        //{
            if(feature[sorted_index[0]] != feature[sorted_index[n-1]]){
                if(min_score>score){
                    min_score = score;
                    best_split_value = split_value;
                    split_feature = i;
                }
            }
            //}
            criterion->reset();
        }
        
        if(any_split && n/total_obs!=1.0 && adaptive_complexity){
            dVector gum_cdf_mmcir_complement = dVector::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
            double expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
            double CRt = - criterion->optimism * (n/total_obs)  *expected_max_S;
            double expected_reduction = criterion->observed_reduction + CRt;
            //printf("%f %f %f %f %f %f\n", expected_reduction, criterion->observed_reduction, CRt, criterion->optimism, total_obs,criterion->node_score);
            if(expected_reduction<0.0){
                any_split = false;
            }
        }
        
        return tuple<bool, int, double, double,double>(any_split, split_feature,impurity,min_score, best_split_value);
        
    
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