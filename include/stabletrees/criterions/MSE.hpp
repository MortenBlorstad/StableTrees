#pragma once
#ifndef __MSE_HPP_INCLUDED__

#define __MSE_HPP_INCLUDED__



#include <Eigen/Dense>
#include "criterion.hpp"
using namespace std;


class MSE : public Criterion{ 
    public:
        void init(double _n, const dVector &y);
        void update(double y_i);
        void reset();
        double node_impurity(const dVector &y);
        double get_score();
    protected:
        double y_sum_squared;
};

double MSE::get_score(){
    return score;
}

void MSE::init(double _n, const dVector &y){
    Criterion::init(_n,y);
    y_sum_squared = y.array().square().sum();
    node_score = node_impurity(y);
    score = 0;
}

double MSE::node_impurity(const dVector &y){
    double pred = y.array().mean();
    return (y.array() - pred).square().mean();
}


    
void MSE::update(double y_i){
    sum_y_l+= y_i;
    sum_y_r-=y_i;
    n_l+=1;
    n_r-=1;
    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    double SSE_L = n_l*( pow(y_bar_l,2.0) );
    double SSE_R = n_r*( pow(y_bar_r,2.0) );
    score = (y_sum_squared - SSE_L-SSE_R)/n;
    //printf("update real %f %f %f %f \n ",SSE_L, SSE_R, n, score);
}

void MSE::reset(){
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_r = n;
    n_l = 0;
}


class MSEReg : public MSE{ 
    public:
        void init(double _n, const dVector &y,const dVector &yprev);
        void update(double y_i, double yp_i);
        void reset();
    protected:
        double y_sum_squared;
};

#endif