#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#include <Eigen/Dense>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dArray = Eigen::Array<double,Eigen::Dynamic,1>;

#include "splitter.hpp"
#include "node.hpp"
#include <stdexcept>



class SplitterReg: public Splitter{
    public:
        SplitterReg();
        SplitterReg(double lambda, int min_samples_leaf,double _total_obs, int _citerion, bool _adaptive_complexity);
        ~SplitterReg();
        tuple<bool, int, double, double,double,double> find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev);
        tuple<bool,double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index,const dVector &y_prev);
    protected:
        Criterion *criterion;
        double total_obs;
};

SplitterReg::SplitterReg(){
    criterion = new MSEReg();
    criterion->set_lambda(0.5);
    adaptive_complexity =false;
    this->min_samples_leaf = 1;
}

SplitterReg::SplitterReg(double lambda, int min_samples_leaf,double _total_obs, int _citerion, bool _adaptive_complexity){
    if(_citerion == 0){
        criterion = new MSEReg();
    }else if(_citerion ==1){
        criterion = new PoissonReg();
    }else{
        throw invalid_argument("Possible criterions are 'mse' and 'poisson'.");
    }
    criterion->set_lambda(lambda);
    total_obs = _total_obs;
    adaptive_complexity =_adaptive_complexity;
    this->min_samples_leaf = min_samples_leaf;
}
SplitterReg::~SplitterReg(){
    delete criterion;
    min_samples_leaf = NULL;
}

tuple<bool,double,double> SplitterReg::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index,const dVector &y_prev){
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
        criterion->update(y[low], y_prev[low]);


        if(lowValue == largestValue){
            break;
        }
        if(hightValue-lowValue<0.0000001){
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
        double score_reg = criterion->get_reg();
        if(min_score>score_reg){
            min_score = score_reg;
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

tuple<bool, int, double, double,double,double>SplitterReg::find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev){
        double min_score = std::numeric_limits<double>::infinity();
        
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        bool any_split_;
        double split_value;
        int i;
        int n = y.size();
        bool any_split = false;
        if(adaptive_complexity){
            grid = dVector::LinSpaced( grid_size, 0.0, grid_end );
            gum_cdf_mmcir_grid = dArray::Ones(grid_size);
        }

        criterion->init(n,y,y_prev);
    
        
        double impurity = criterion->node_score;

     
        iMatrix X_sorted_indices = sorted_indices(X);
    
        


        //#pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature) private(i,score,split_value, feature)
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
          
            tie(any_split_, score, split_value) = select_split_from_all(feature, y, sorted_index, y_prev);
            
            
           //#pragma omp ordered
        //{
            
            if(feature[sorted_index[0]] != feature[sorted_index[n-1]]){
                if(any_split_ && min_score>score){
                    any_split = true;
                    min_score = score;
                    best_split_value = split_value;
                    split_feature = i;
                }
            }
            //}
            criterion->reset();
        }
        //&& n/total_obs!=1.0
        double w_var = 0.0;
        if(any_split && n/total_obs!=1.0  && adaptive_complexity){
            
            dVector gum_cdf_mmcir_complement = dVector::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
            double expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
            double CRt = criterion->optimism * (n/total_obs)  *expected_max_S;
            //double C = CRt*(criterion->n_r/criterion->n) + CRt*(criterion->n_r/criterion->n);
            double expected_reduction = 1.0*(2.0-1.0)*criterion->observed_reduction*((n/total_obs) ) - 1.0*CRt;
            w_var = criterion->optimism/(criterion->H/criterion->n);
            // std::cout << "local_optimism: " <<  criterion->optimism<< std::endl;
            // std::cout << "CRt: " <<  CRt << std::endl;
            // std::cout << "n:  " <<  n  <<std::endl;
            // std::cout << "prob_node:  " <<  n/total_obs << std::endl;
            // std::cout << "expected_max_S:  " <<  expected_max_S << std::endl;
            // std::cout << "observed_reduction:  " <<  criterion->observed_reduction << std::endl;
            // std::cout << "expected_reduction:  " <<  expected_reduction << "\n" <<std::endl;
           
           if(expected_reduction<0.0){
                any_split = false;
            }
        }
        
        return tuple<bool, int, double, double,double,double>(any_split, split_feature,impurity,min_score, best_split_value,w_var);
        
}
#endif