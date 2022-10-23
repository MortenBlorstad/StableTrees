#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#define __SLITTER_HPP_INCLUDED__

#include <Eigen/Dense>


using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;


using namespace std;



class Splitter{

    public:
        Splitter(); 
        string type= "normal";
        // double get_split( dMatrix &X, dVector &y);
        double sum_squared_error(dVector &y_true, double  y_pred);
        tuple<double,double,dVector,dVector > get_predictions(dVector &feature, dVector &y, double value);
        double mse_criterion(dVector  &feature,dVector  &y, double  value);
        tuple<double,double> select_split(dVector  &feature, dVector  &y);
        tuple<int,double,double>  find_best_split(dMatrix  &X, dVector  &y);

    //private:
        //double mse_criterion(dMatrix &feature, dVector &y, bVector &mask);
        //tuple<double,double> select_split(dVector&feature, dVector &y);
        //tuple<int,double,double> select_split(dMatrix&X, dVector &y);

};

Splitter::Splitter(){
    this->type = "normal"; 
    std::vector<int> mask;  
}

tuple<double,double, dVector, dVector> Splitter::get_predictions(dVector &feature, dVector &y, double value){
    double left_prediction = 0.0;
    double right_prediction = 0.0;
    std::vector<double> left_values;
    std::vector<double> right_values;
    for(int i=0; i<y.rows();i++){
        if(feature[i]<=value){
            left_prediction+=y[i];
            left_values.push_back(y[i]);
        }else{
            right_prediction+=y[i];
            right_values.push_back(y[i]);
        }
    
    }
    if(left_values.size()>0){
        left_prediction/=left_values.size();
    }
    if(right_values.size()>0){
        right_prediction/=right_values.size();
    }

    dVector left_values_v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(left_values.data(), left_values.size());
    dVector right_values_v = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(right_values.data(), right_values.size());

    return tuple<double,double,dVector, dVector>(left_prediction,right_prediction, left_values_v, right_values_v);
}

double Splitter::sum_squared_error(dVector &y_true, double  y_pred){
    
    return (y_true.array() - y_pred).pow(2.0).sum();

}

double Splitter::mse_criterion(dVector  &feature,dVector  &y, double  value){

    double left_pred;double right_pred; dVector left_values; dVector right_values;
    tie(left_pred, right_pred, left_values, right_values) = get_predictions(feature, y, value);
    double left = 0.0;
    double right = 0.0;
    if(left_values.size() >0){

        left = sum_squared_error(left_values,left_pred);

    }
    if (right_values.size()>0){

        right = sum_squared_error( right_values ,right_pred);

    }
    


    return (right+left)/2;

}

tuple<double,double> Splitter::select_split(dVector  &feature, dVector  &y){
    double min_score = INFINITY;
    double score;
    double split_value = feature.mean();
    score = mse_criterion(feature, y, split_value);
    return tuple<double,double>(score,split_value);
}

tuple<int, double,double> Splitter::find_best_split(dMatrix  &X, dVector  &y){
    double min_score = std::numeric_limits<double>::infinity();
    double best_split_value;
    double score;
    double split_value;
    int split_feature;
    dVector feature; 
    for(int i =0; i<X.cols(); i++){
        feature = X.col(i);
        tie(score, split_value) = select_split(feature, y);
        if(min_score>score){
            min_score = score;
            best_split_value = split_value;
            split_feature = i;
        }
    }
    return tuple<int, double,double>(split_feature,min_score, best_split_value);
}


#endif