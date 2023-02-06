#pragma once
#ifndef __POISSON_HPP_INCLUDED__

#define __POISSON_HPP_INCLUDED__



#include <Eigen/Dense>
#include "criterion.hpp"
using namespace std;


class Poisson : public Criterion{ 
    public:
        ~Poisson();
        void init(double _n, const dVector &y);
        void update(double y_i);
        void reset();
        double node_impurity(const dVector &y);
        double get_score();
        bool should_skip(int min_samples_leaf);
    protected:
        double sum_ylogy_l;
        double sum_ylogy_r;
        double sum_ylogy;

};

Poisson::~Poisson(){
    Criterion::~Criterion();
    sum_ylogy_l = NULL;
    sum_ylogy_r = NULL;
    sum_ylogy = NULL;
}


double Poisson::get_score(){
    return score;
}

void Poisson::init(double _n, const dVector &y){
    Criterion::init(_n,y);
    sum_ylogy_l = 0;
    sum_ylogy = (y.array()*log(y.array()+ eps)).sum();
    sum_ylogy_r = sum_ylogy;

    node_score = node_impurity(y);
    score = 0;
}

double Poisson::node_impurity(const dVector &y){
    double pred = y.array().mean();
    return 2*((y.array()+eps)*log((y.array()+eps)/(pred+eps)) - (y.array()-pred)).mean();
}

bool Poisson::should_skip(int min_samples_leaf){
    bool skip = false;
    if(Criterion::should_skip(min_samples_leaf)){
        skip = true;
    }
    if(sum_y_l<=eps || sum_y_r<=eps){
        skip = true;
    }
    
    return skip;
}
    
void Poisson::update(double y_i){
    Criterion::update(y_i);
    
    sum_ylogy_l+= (y_i+eps)*log(y_i+eps);
    sum_ylogy_r-= (y_i+eps)*log(y_i+eps);

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;

    double sum_ylogpred_l = sum_y_l*log(y_bar_l+eps);
    double sum_ylogpred_r = sum_y_r*log(y_bar_r+eps);
    score = (sum_ylogy_l-sum_ylogpred_l + sum_ylogy_r - sum_ylogpred_r)/n;
}

void Poisson::reset(){
    Criterion::reset();
    sum_ylogy_l = 0;
    sum_ylogy_r = sum_ylogy;

}

class PoissonReg : public Poisson{ 
    public:
        void init(double _n, const dVector &y, const dVector &yprev);
        void update(double y_i,double yp_i);
        void reset();
        ~PoissonReg();

    protected:
        double sum_prev_ylogy_l;
        double sum_prev_ylogy_r;
        double sum_prev_ylogy;
        double sum_yprev_l;
        double sum_yprev_r;
        double sum_yprev;

};

PoissonReg::~PoissonReg(){
    Poisson::~Poisson();
    sum_prev_ylogy_l = NULL;
    sum_prev_ylogy_r = NULL;
    sum_prev_ylogy = NULL;
    sum_yprev_l = NULL;
    sum_yprev_r = NULL;
    sum_yprev = NULL;
}
void PoissonReg::init(double _n, const dVector &y, const dVector &yprev){
    Poisson::init(_n,y);
    sum_prev_ylogy = (yprev.array()*log(yprev.array()+ eps)).sum();
    sum_prev_ylogy_l = 0;
    sum_prev_ylogy_r = sum_prev_ylogy;

    sum_yprev = yprev.array().sum();
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;
}


void PoissonReg::reset(){
    Poisson::reset();
    sum_prev_ylogy_l = 0;
    sum_prev_ylogy_r = sum_prev_ylogy;
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;
}

void PoissonReg::update(double y_i,double yp_i){
    Poisson::update(y_i);
    sum_yprev_l+=yp_i;
    sum_yprev_r-=yp_i;
    sum_prev_ylogy_l += (yp_i+eps)*log(yp_i+eps);
    sum_prev_ylogy_r -= (yp_i+eps)*log(yp_i+eps);

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    double sum_prev_ylogpred_l = sum_yprev_l*log(y_bar_l+eps);
    double sum_prev_ylogpred_r = sum_yprev_r*log(y_bar_r+eps);
   

    double reg = ( (sum_prev_ylogy_l - sum_prev_ylogpred_l - (sum_yprev_l - n_l*y_bar_l) ) + (sum_prev_ylogy_r - sum_prev_ylogpred_r  - (sum_yprev_r - n_r*y_bar_r) ) )/n;
    score+=reg;
}


#endif