#pragma once
#ifndef __Splitter_HPP_INCLUDED__
#pragma once
#include "splitter.hpp"
#include "node.hpp"
#include <stdexcept>

class SplitterReg: public Splitter{
    public:
        SplitterReg();
        tuple<int,double,double>  find_best_split(const dMatrix  &X, const dVector  &y,const dVector &y_prev);
        tuple<double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index,const dVector &y_prev);

};

SplitterReg::SplitterReg():Splitter(){
    Splitter();
}






tuple<double,double> SplitterReg::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index,const dVector &y_prev){
    //https://stats.stackexchange.com/questions/72212/updating-variance-of-a-dataset
    double min_score = std::numeric_limits<double>::infinity();
    double n = y.size();
    double best_split_value;
    
    double y_L = 0;
    double y_R = y.array().sum();
    double N_L = 0;
    double N_R = n;
    double y_squared = y.array().square().sum();

    double largestValue = feature(sorted_index[n-1]);


    double yp_L = 0;
    double yp_R = y_prev.array().sum();
    double yp_squared = y_prev.array().square().sum();


    for (int i = 0; i < n-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];
        
        double split_value =  (lowValue+hightValue)/2;
        y_L += y(low);
        y_R -= y(low);
        yp_L += y_prev(low);
        yp_R -= y_prev(low);
        N_L+=1;
        N_R-=1;
        if(hightValue-lowValue<0.0001){
            continue;
        }
        if(lowValue == largestValue){
            break;
        }
            
        double y_L_bar = y_L/N_L;
        double y_R_bar = y_R/N_R;
        double SSE_L= N_L*pow(y_L_bar,2);
        double SSE_R= N_R*pow(y_R_bar,2);
        double score = (y_squared - SSE_L - SSE_R)/n;
        double R = (SSE_L+ SSE_R - 2*yp_L*y_L_bar - 2*yp_R*y_R_bar  + yp_squared) /n;
        score = score + R;

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
        

        iMatrix X_sorted_indices = sorted_indices(X);
        


        #pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature) private(i,score,split_value, feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
            tie(score, split_value) = select_split_from_all(feature, y, sorted_index,y_prev);
            //printf("num obs %d,split_value %f , min_score %f \n", feature.size(),split_value,min_score);
           #pragma omp ordered
        {
            if(feature[sorted_index[0]] != feature[sorted_index[feature.rows()-1]]){
               //printf("%d, %f \n",i,score);
                if(min_score>score){
                    min_score = score;
                    best_split_value = split_value;
                    split_feature = i;
                }
            }
            }
        }
        //printf("=== %d, %f, %f  \n ",split_feature,min_score, best_split_value);
        return tuple<int, double,double>(split_feature,min_score, best_split_value);
    
    
}
#endif