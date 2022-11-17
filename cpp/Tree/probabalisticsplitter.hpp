#pragma once
#ifndef __ProbabalisticSplitter_HPP_INCLUDED__

#define __ProbabalisticSplitter_HPP_INCLUDED__


#include <Eigen/Dense>
#include <unordered_set>
#include "splitter.hpp"
#include "node.hpp"
#include <omp.h>
#include <random>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;


using namespace std;



class ProbabalisticSplitter: public Splitter{

    public:
        ProbabalisticSplitter::ProbabalisticSplitter(int seed);
        virtual tuple<vector<tuple<int, double,double>>,double>   select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index, int feature_index);
        virtual tuple<int, double,double> find_best_split(const dMatrix  &X, const dVector  &y);
    private:
        int seed;
        
    
};

ProbabalisticSplitter::ProbabalisticSplitter(int seed):Splitter(){
    Splitter();
    this->seed = seed;
}

tuple<vector<tuple<int,double,double>>,double> ProbabalisticSplitter::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index, int feature_index){
    float n = y.size();


    double y_L = 0;
    double y_R = y.array().sum();
    float N_L = 0;
    float N_R = n;
    double y_squared = y.array().square().sum();
    double total = 0;
    vector<tuple<int, double,double>> splits(y.size()-1);
    for (int i = 0; i < sorted_index.rows()-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];


        double split_value =  (lowValue+hightValue)/2;
        y_L+= y(low);
        y_R-= y(low);
        N_L+=1;
        N_R-=1;
        

        double SSE_L= N_L*pow((y_L/N_L),2);
        double SSE_R= N_R*pow((y_R/N_R),2);
        double score = y_squared - SSE_L - SSE_R;

        total+=pow(1/score,2.0);
        
        splits[i] = tuple<int, double,double>(feature_index, score, split_value);
        
        

    }
  
    
    return tuple<vector<tuple<int, double,double>>,double>(splits,total);
}



tuple<int, double,double> ProbabalisticSplitter::find_best_split(const dMatrix  &X, const dVector  &y){
    
    
        dVector feature;
        iMatrix X_sorted_indices = sorted_indices(X);
        

        vector<tuple<int, double,double>> all_splits((y.size()-1)*X.cols());

        double total = 0;

       // #pragma omp parallel for ordered num_threads(4) shared(total,all_splits) private(feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            double part_total = 0;
            vector<tuple<int,double,double>> splits(y.size()-1); 
            iVector sorted_index = X_sorted_indices.col(i);
            
           
            tie(splits,part_total) = select_split_from_all(feature, y, sorted_index, i);
           
            
            //#pragma omp ordered
            //{  
                total += part_total;
                for(int j = 0; j<splits.size();j++){
                    
                    all_splits[j+i*splits.size()] = splits[j];
                }
            
           // }
        }
        vector<double> chances(all_splits.size());
        
        for(int i = 0; i<all_splits.size();i++){
            chances[i] =  exp(get<1>(all_splits[i]));
        }
        
        std::mt19937 gen(this->seed );
        std::discrete_distribution<std::size_t> d{chances.begin(), chances.end()};
        size_t ind = d(gen);
        tuple<int, double,double> sampled_value = all_splits[ind];
        

        return sampled_value;
    
    
}


#endif