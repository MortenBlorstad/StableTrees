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
    score = 0;
    node_score = (y_sum_squared - n*(pred*pred))/n;
    G = 2*(pred - y.array()).sum();
    H = 2*n;
    G_l = 0; H_l = 0;
    G2 = (2*(pred - y.array()).square()).sum();
    H2 = 4*n; gxh = (2*(pred - y.array())*2).sum();
    optimism = (G2 + 2*gxh*(pred) + H2*pred*pred)/(n*H);

}

double MSE::node_impurity(const dVector &y){
    if(node_score >=0.0){
        return node_score;
    }
    double pred = y.array().mean();
    node_score = (y.array() - pred).square().mean();
    return node_score;
}


    
void MSE::update(double y_i){
    Criterion::update(y_i);
    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    double SSE_L = n_l*( pow(y_bar_l,2.0) );
    double SSE_R = n_r*( pow(y_bar_r,2.0) );
    G_l += 2*(pred  -  y_i); H_l += 2;
    double G_r = G - G_l; double H_r = H-H_l;

    //score = -(G_l*G_l/H_l + G_r*G_r/H_r - G*G/H)/(2*n);
    score = (y_sum_squared - SSE_L-SSE_R)/n;
    //printf("update real %f %f %f %f \n ",SSE_L, SSE_R, n, score);
}

void MSE::reset(){
    Criterion::reset();
}


class MSEReg : public MSE{ 
    public:
        void init(double _n, const dVector &y,const dVector &yprev);
        void update(double y_i, double yp_i);
        void reset();
    protected:
        double yprev_sum_squared;
        double sum_yprev_l;
        double sum_yprev_r;
        double sum_yprev;
};


void MSEReg::reset(){
    MSE::reset();
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;  
}




void MSEReg::init(double _n, const dVector &y,const dVector &yprev){
    MSE::init(_n,y);
    yprev_sum_squared = yprev.array().square().sum();
    sum_yprev = yprev.array().sum();
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;
    
    // node_score += (yprev_sum_squared - n*(pred*pred))/n;

    // G += 2*(pred - yprev.array()).sum();
    // H += 2*n;
    // G2 += (2*(pred - yprev.array()).square()).sum();
    // H2 += 4*n; gxh += (2*(pred - yprev.array())*2).sum();
    // optimism += (G2 + 2*gxh*(pred) + H2*pred*pred)/(n*H);
}

void MSEReg::update(double y_i, double yp_i){
    MSE::update(y_i);
    sum_yprev_l+= yp_i;
    sum_yprev_r-=yp_i;

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    // G_l += 2*(y_bar_l  -  yp_i); H_l += 2;
    // double G_r = G - G_l; double H_r = H-H_l;

    double reg = (n_l*pow(y_bar_l,2.0) + n_r*pow(y_bar_r,2.0) - 2*sum_yprev_l*y_bar_l - 2*sum_yprev_r*y_bar_r + yprev_sum_squared)/n;
    score += reg;
}

#endif