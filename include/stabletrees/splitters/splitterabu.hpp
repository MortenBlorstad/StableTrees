#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#include "splitter.hpp"
#include "node.hpp"
#include <stdexcept>

class SplitterABU: public Splitter{
    public:
        SplitterABU();
        SplitterABU(int min_samples_leaf, int _citerion);
        SplitterABU(int max_depth,double min_split_sample, int min_samples_leaf, double _total_obs, int _citerion);
        ~SplitterABU();
        tuple<bool, int,double, double,double,double,double>  find_best_split(const dMatrix  &X, const dVector  &y,const dMatrix  &X_sample, dMatrix &leaf_info, int depth);
        tuple<bool, int,double, double,double,double,double>  find_best_split(const dMatrix  &X, const dVector  &y, int depth);
        tuple<bool, double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector &sorted_index,const dVector  &feature_sample, const iVector &sorted_sample_index, const dMatrix &leaf_info);
        tuple<bool, double,double> select_split_from_all(const dVector  &feature, const dVector  &y,const iVector &sorted_index);
        dMatrix cir_sim;
        double grid_end; 
    protected:
        Criterion *criterion;
        Criterion *criterion_update;
        int grid_size = 101;
        dVector grid;
        int max_depth;
        double min_split_sample;
        bool all_same_features_values(const dMatrix &X);
        bool all_same(const dVector &vec);
        tuple<bool, double> stop_criteria(const dMatrix  &X, const dVector  &y,int depth);
        tuple<iVector, iVector> get_masks(const dVector &feature, double value);
};

tuple<iVector, iVector> SplitterABU::get_masks(const dVector &feature, double value){
    std::vector<int> left_values;
    std::vector<int> right_values;
    for(int i=0; i<feature.rows();i++){
        if(feature[i]<=value){
            left_values.push_back(i);
        }else{
            right_values.push_back(i);
        }
    }
    iVector left_values_v = Eigen::Map<iVector, Eigen::Unaligned>(left_values.data(), left_values.size());
    iVector right_values_v = Eigen::Map<iVector, Eigen::Unaligned>(right_values.data(), right_values.size());

    return tuple<iVector, iVector> (left_values_v, right_values_v);
}

SplitterABU::SplitterABU(){
    criterion = new MSE();
    criterion_update = new MSEABU(min_samples_leaf);
    this->min_samples_leaf = 2;
    this->max_depth = INT_MAX;
    min_split_sample = 2;
    set_seed(1);
    cir_sim = cir_sim_mat(100, 100);
    grid_end = 1.5*cir_sim.maxCoeff();
}
SplitterABU::SplitterABU(int max_depth,double min_split_sample, int min_samples_leaf, double _total_obs, int _citerion){
    this->max_depth = max_depth;
    if(_citerion == 0){
        criterion = new MSEABU(min_samples_leaf);
        criterion_update = new MSEABU(min_samples_leaf);
    }else if(_citerion ==1){
        criterion = new Poisson();
    }else{
        throw invalid_argument("Possible criterions are 'mse' and 'poisson'.");
    }
    total_obs = _total_obs;
    this->min_samples_leaf = min_samples_leaf;
    this->max_depth = max_depth;
    this->min_split_sample = min_split_sample;
    set_seed(1);
    cir_sim = cir_sim_mat(100, 100);
    grid_end = 1.5*cir_sim.maxCoeff();
}


SplitterABU::~SplitterABU(){
    delete criterion;
    min_samples_leaf = NULL;
}

bool SplitterABU::all_same(const dVector &vec){
    bool same = true;
    for(int i=0; i< vec.rows(); i++){
        if(vec(i)!=vec(0) ){
            same=false;
            break;
        }
    }
    return same;
}

bool SplitterABU::all_same_features_values(const dMatrix &X){
    bool same = true;
    dVector feature;
    for(int i =0; i<X.cols(); i++){
        feature = X.col(i);
        if(!all_same(feature)){
            same=false;
            break;
        }
    }
    return same;
}

tuple<bool, double,double> SplitterABU::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector &sorted_index,const dVector  &feature_sample,
                                                                 const iVector &sorted_sample_index,const dMatrix &leaf_info){
     //https://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset
    double min_score = std::numeric_limits<double>::infinity();
    double min_score_reg = std::numeric_limits<double>::infinity();
    double n = y.size();
    double best_split_value;
    
    int num_splits = 0;
    double largestValue = feature(sorted_index[n-1]);
    dVector u_store((int)n);
    u_store = dVector::Zero(n);

    double prob_delta = 1.0/n;
    dArray gum_cdf_grid(grid_size);

    
    bool any_split = false;

    for (int i = 0; i < n-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];
        
        double split_value =  (lowValue+hightValue)/2;
        //printf("before update\n");
        criterion_update->update(y[low],sorted_sample_index,feature_sample, leaf_info,split_value);
        //printf("after update\n");
        // iVector left;iVector right;
        // iVector keep_cols = iVector::LinSpaced(leaf_info.cols(), 0, leaf_info.cols()-1).array();
        // tie(left,right) = get_masks(feature_sample,split_value);
        // dVector info = leaf_info(left,keep_cols).colwise().mean();

        //printf("control left %f, %f, %f, %d\n",info(1,0)/info(2,0)/(double)left.size(), info(1,0), info(2,0), left.size());
    
        // break if rest of the values are equal
        if(lowValue == largestValue){
            break;
        }
        // skip if values are approx equalprintf("
        if(hightValue-lowValue<0.00000000001){
            continue;
        }
        if(criterion_update->should_skip(min_samples_leaf)){
            continue;
        }
        
        u_store[num_splits] = (i+1)*prob_delta;
        num_splits +=1;
        any_split = true;
        double score = criterion_update->get_score();
        double score_reg = criterion_update->score_reg;

        //printf("score_reg %f, score  %f\n", score_reg, score);
        
        if(min_score_reg>score_reg){
            //printf("update score %f %f\n", score,score_reg);
            min_score = score;
            min_score_reg = score_reg;
            best_split_value = split_value;
            criterion_update->observed_reduction = criterion_update->node_score - score;
        }
    }
    if(num_splits<=0){
        any_split = false;
    }else{
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
    
    return tuple<bool,double,double>(any_split,min_score,best_split_value);
}

tuple<bool, double,double> SplitterABU::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector &sorted_index){
    //https://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset
    double min_score = std::numeric_limits<double>::infinity();
    double n = y.size();
    double best_split_value;
    
    int num_splits = 0;
    double largestValue = feature(sorted_index[n-1]);
    dVector u_store((int)n);
    u_store = dVector::Zero(n);

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
        
    
        // break if rest of the values are equal
        if(lowValue == largestValue){
            break;
        }
        // skip if values are approx equal
        if(hightValue-lowValue<0.00000000001){
            continue;
        }
        if(criterion->should_skip(min_samples_leaf)){
            continue;
        }
        u_store[num_splits] = (i+1)*prob_delta;
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
    }else{
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
    
    return tuple<bool,double,double>(any_split,min_score,best_split_value);
}
tuple<bool, double> SplitterABU::stop_criteria(const dMatrix  &X, const dVector  &y,int depth){
    double pred = y.array().mean();
    double y_var = 0.0;
   
    bool any_split = false;
    if (depth>= this->max_depth){
            y_var = (y.array() - pred).square().mean();
            return tuple<bool,double>(any_split,y_var);   
    }
    if(X.rows()<2 || y.rows()<2){
        return tuple<bool, double> (any_split,0.0); 
    }
    if(y.rows()< this->min_split_sample){
        any_split = false;
        
        y_var = (y.array() - pred).square().mean();
        
        return tuple<bool,double>(any_split,y_var); 
    }
    if(all_same(y)){
        return tuple<bool,double>(any_split, 0.0); 
    }
    if(all_same_features_values(X)){
        double y_var = 0.0;
        double pred = y.array().mean();
        y_var = (y.array() - pred).square().mean();
        return tuple<bool,double>(any_split,y_var); 
    } 
    y_var = (y.array() - pred).square().mean();
    any_split = true;
    return tuple<bool,double>(any_split,y_var);

}

tuple<bool, int, double, double,double,double,double> SplitterABU::find_best_split(const dMatrix  &X, const dVector  &y,const dMatrix  &X_sample, dMatrix &leaf_info, int depth){
        double min_score = std::numeric_limits<double>::infinity();
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        bool any_split;
        bool found_any_split;
        double split_value = false;
        grid = dVector::LinSpaced( grid_size, 0.0, grid_end );
        gum_cdf_mmcir_grid = dArray::Ones(grid_size);
        int i;
        int n = y.size();
        
        iMatrix X_sorted_indices = sorted_indices(X);
        iMatrix sorted_indices_sample = sorted_indices(X_sample);

        criterion_update->init(n,y,leaf_info);
        double impurity = criterion_update->node_score;
        double y_var;
        //printf("before stop_criteria\n");
        tie(any_split,y_var) = stop_criteria(X,y,depth);
        //printf("after stop_criteria\n");
        criterion_update->reset();
        if(any_split){
            any_split = false;
            for(int i =0; i<X.cols(); i++){
                feature = X.col(i);
                dVector feature_sample = X_sample.col(i);
                iVector sorted_index = X_sorted_indices.col(i);
                iVector sorted_index_sample = sorted_indices_sample.col(i);
                //printf("before %d\n", i);
                tie(found_any_split, score, split_value) = select_split_from_all(feature, y, sorted_index,feature_sample,sorted_index_sample,leaf_info);
                    //printf("after %d\n", i);
                if(feature[sorted_index[0]] != feature[sorted_index[n-1]]){
                    if(found_any_split && min_score>score){
                        min_score = score;
                        best_split_value = split_value;
                        split_feature = i;
                        any_split = true;
                    }
                }
            //printf("\n");
            criterion_update->reset();
            }
        }
        //printf("after loop\n");
        dVector gum_cdf_mmcir_complement = dVector::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
        double expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
        double CRt = criterion_update->optimism * (n/total_obs)  *expected_max_S;
        //double C = CRt*(criterion->n_r/criterion->n) + CRt*(criterion->n_r/criterion->n);
        double expected_reduction = 1.0*(2.0-1.0)*criterion_update->observed_reduction*((n/total_obs) ) - 1.0*CRt;
        double w_var = (n/total_obs)*(criterion_update->optimism/(criterion_update->H/n)); //criterion->optimism/(criterion->H/criterion->n); 
        
        // std::cout << "update: " << std::endl;
        // std::cout << "local_optimism: " <<  criterion_update->optimism<< std::endl;
        // std::cout << "CRt: " <<  CRt << std::endl;
        // std::cout << "n:  " <<  n  <<std::endl;
        // std::cout << "prob_node:  " <<  n/total_obs << std::endl;
        // std::cout << "expected_max_S:  " <<  expected_max_S << std::endl;
        // std::cout << "node_score:  " <<  criterion_update->node_score << std::endl;
        // std::cout << "observed_reduction:  " <<  criterion_update->observed_reduction << std::endl;
        // std::cout << "expected_reduction:  " <<  expected_reduction <<std::endl;
        // std::cout << "H:  " <<  criterion_update->H  <<std::endl;
        // std::cout << "y_var:  " <<  y_var <<std::endl;
        // std::cout << "w_var:  " <<  w_var <<std::endl;
        //printf("%f %f %f %f %f %f %f %f \n", expected_reduction, criterion->observed_reduction, CRt,C, criterion->optimism, n/total_obs, criterion->node_score,expected_max_S);
        if(any_split && n/total_obs!=1.0 && expected_reduction<0.0){
            any_split = false;
        }
        //std::cout << "any_split:  " <<  any_split << "\n" <<std::endl;
        
        return tuple<bool, int, double, double, double,double,double>(any_split, split_feature,impurity,min_score, best_split_value,y_var,w_var);
    
    
}






tuple<bool, int, double, double,double,double,double> SplitterABU::find_best_split(const dMatrix  &X, const dVector  &y, int depth){
        double min_score = std::numeric_limits<double>::infinity();
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        bool any_split;
        bool found_any_split;
        double split_value = false;
        grid = dVector::LinSpaced( grid_size, 0.0, grid_end );
        gum_cdf_mmcir_grid = dArray::Ones(grid_size);
        int i;
        int n = y.size();
        iMatrix X_sorted_indices = sorted_indices(X);
 
        criterion->init(n,y);

        double impurity = criterion->node_score;
        
        double y_var;
   
        tie(any_split,y_var) = stop_criteria(X,y,depth);

        
        if(any_split){
            for(int i =0; i<X.cols(); i++){
                any_split = false;
                feature = X.col(i);
                iVector sorted_index = X_sorted_indices.col(i);
                tie(found_any_split, score, split_value) = select_split_from_all(feature, y, sorted_index);
                
        
                if(feature[sorted_index[0]] != feature[sorted_index[n-1]]){
                    if(found_any_split && min_score>score){
                        min_score = score;
                        best_split_value = split_value;
                        split_feature = i;
                        any_split = true;
                    }
                }
    
                criterion->reset();
            }

        }
        
        
        dVector gum_cdf_mmcir_complement = dVector::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
        double expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
        double CRt = criterion->optimism * (n/total_obs)  *expected_max_S;
        //double C = CRt*(criterion->n_r/criterion->n) + CRt*(criterion->n_r/criterion->n);
        double expected_reduction = 1.0*(2.0-1.0)*criterion->observed_reduction*((n/total_obs) ) - 1.0*CRt;
        double w_var = (n/total_obs)*(criterion->optimism/(criterion->H/n)); //criterion->optimism/(criterion->H/criterion->n); 
        // std::cout << "learn: " << std::endl;
        // std::cout << "local_optimism: " <<  criterion->optimism<< std::endl;
        // std::cout << "CRt: " <<  CRt << std::endl;
        // std::cout << "n:  " <<  n  <<std::endl;
        // std::cout << "prob_node:  " <<  n/total_obs << std::endl;
        // std::cout << "expected_max_S:  " <<  expected_max_S << std::endl;
        // std::cout << "observed_reduction:  " <<  criterion->observed_reduction << std::endl;
        // std::cout << "expected_reduction:  " <<  expected_reduction <<std::endl;
        // std::cout << "H:  " <<  criterion->H  <<std::endl;
        // std::cout << "y_var:  " <<  y_var <<std::endl;
        // std::cout << "w_var:  " <<  w_var << "\n" <<std::endl;
        
        //printf("%f %f %f %f %f %f %f %f \n", expected_reduction, criterion->observed_reduction, CRt,C, criterion->optimism, n/total_obs, criterion->node_score,expected_max_S);
            
        
        if(any_split && n/total_obs!=1.0 && expected_reduction<0.0){
                any_split = false;
        }
        return tuple<bool, int, double, double, double,double,double>(any_split, split_feature,impurity, min_score, best_split_value,y_var, w_var);      
}
#endif