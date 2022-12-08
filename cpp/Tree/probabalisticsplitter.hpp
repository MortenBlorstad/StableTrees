#pragma once
#ifndef __ProbabalisticSplitter_HPP_INCLUDED__

#define __ProbabalisticSplitter_HPP_INCLUDED__


#include <Eigen/Dense>
#include <unordered_set>
#include "splitterReg.hpp"
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



class ProbabalisticSplitter: public SplitterReg{

    public:
        ProbabalisticSplitter::ProbabalisticSplitter(int seed);
        virtual tuple<int, double,double> find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev);
    private:
        int seed;
        
    
};

ProbabalisticSplitter::ProbabalisticSplitter(int seed):SplitterReg(){
    SplitterReg();
    this->seed = seed;
}



tuple<int, double,double> ProbabalisticSplitter::find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev){
    
    
        dVector feature;
        iMatrix X_sorted_indices = sorted_indices(X);
        

        vector<tuple<int, double,double>> all_splits;

        double score;
        double split_value;
        int i;

       #pragma omp parallel for ordered num_threads(4) shared(all_splits) private(i,score,split_value, feature)

        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
            
            
            tie(score, split_value) = select_split_from_all(feature, y, sorted_index, y_prev);
            //printf("%d, %f, %f \n", i,score, split_value);
            
            #pragma omp ordered
            {  
                if(feature[sorted_index[0]] != feature[sorted_index[feature.rows()-1]] && score < std::numeric_limits<double>::infinity()){
                    
                    all_splits.push_back(tuple<int, double,double>(i,score,split_value));
                }
            
            }
        }
        vector<double> chances(all_splits.size());
        
        double total = 0;
        for(int i = 0; i<all_splits.size();i++){
            total +=1/pow(get<1>(all_splits[i]),4.0);
        }
        for(int i = 0; i<all_splits.size();i++){
            double chance = (1/pow(get<1>(all_splits[i]),4.0))/total;
            
            chances[i] =  chance;
        }
        std::mt19937 gen(this->seed);
        std::discrete_distribution<std::size_t> d{chances.begin(), chances.end()};
        size_t ind = d(gen);
        auto&  sampled_value = all_splits.at(ind);
        tie(i, score, split_value) = sampled_value;

        /*for (auto& tup : all_splits) {
            cout << get<1>(tup) << endl;
        }*/
        //printf("%d, %f, %f \n", get<0>(sampled_value),get<1>(sampled_value),get<2>(sampled_value) );
        return tuple<int, double,double>(i,score,split_value);
    
    
}


#endif