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
        
        ProbabalisticSplitter();
        ProbabalisticSplitter(int min_samples_leaf, double _total_obs, bool _adaptive_complexity, int seed);
        ProbabalisticSplitter(int min_samples_leaf,double _total_obs, int _citerion, bool _adaptive_complexity,int seed);
        ~ProbabalisticSplitter();
        virtual tuple<bool, int,double, double,double,double> find_best_split(const dMatrix  &X, const dVector  &y);
    private:
        int seed;
        
    
};

ProbabalisticSplitter::ProbabalisticSplitter():Splitter(){
    Splitter();
    this->seed = 0;
}

ProbabalisticSplitter::~ProbabalisticSplitter(){
    Splitter::~Splitter();
    this->seed = NULL;
}

ProbabalisticSplitter::ProbabalisticSplitter(int min_samples_leaf,double _total_obs, bool _adaptive_complexity,int seed):Splitter(min_samples_leaf,_total_obs,_adaptive_complexity){
    Splitter(min_samples_leaf,_total_obs,_adaptive_complexity);
    this->seed = seed;
}

ProbabalisticSplitter::ProbabalisticSplitter(int min_samples_leaf,double _total_obs, int _citerion, bool _adaptive_complexity,int seed):Splitter(min_samples_leaf,_total_obs,_citerion,_adaptive_complexity){
    Splitter(min_samples_leaf,_total_obs,_citerion,_adaptive_complexity);
    this->seed = seed;
}

tuple<bool, int, double,double, double,double> ProbabalisticSplitter::find_best_split(const dMatrix  &X, const dVector  &y){
    
        dVector feature;
        iMatrix X_sorted_indices = sorted_indices(X);
        

        vector<tuple<int, double,double>> all_splits;

        double score;
        bool any_split;
        double split_value;
        
        int n = y.size();
        criterion->init(n,y);
        

        double impurity = criterion->node_score;

        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
            
            
            tie(any_split,score, split_value) = select_split_from_all(feature, y, sorted_index);
            //printf("%d, %f, %f \n", i,score, split_value);
            
            
            if(feature[sorted_index[0]] != feature[sorted_index[n-1]] && any_split){
                
                all_splits.push_back(tuple<int, double,double>(i,score,split_value));
            }
            criterion->reset();
       
        }
        vector<double> chances(all_splits.size());
        
        double total = 0;
        for(int i = 0; i<all_splits.size();i++){
            total +=1/pow(get<1>(all_splits[i]),2.0);
        }
        for(int i = 0; i<all_splits.size();i++){
            double chance = (1/pow(get<1>(all_splits[i]),2.0))/total;
            
            chances[i] =  chance;
        }
        
        std::mt19937 gen(seed);
        seed = 36969*(seed & 0177777) + (seed>>16) + 1;
        std::discrete_distribution<std::size_t> d{chances.begin(), chances.end()};
        size_t ind = d(gen);
        auto&  sampled_value = all_splits.at(ind);
        int split_feature;
        tie(split_feature, score, split_value) = sampled_value;

        /*for (auto& tup : all_splits) {
            cout << get<1>(tup) << endl;
        }*/

        //printf("%d, %f, %f \n", get<0>(sampled_value),get<1>(sampled_value),get<2>(sampled_value) );
        return tuple<bool, int, double,double,double,double>(any_split, split_feature, impurity, score,split_value,0.0);
}

#endif