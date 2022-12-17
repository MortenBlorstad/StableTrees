#pragma once
#ifndef __POISSON_HPP_INCLUDED__

#define __POISSON_HPP_INCLUDED__



#include <Eigen/Dense>
#include "criterion.hpp"
using namespace std;


class Poisson : public Criterion{ 
    public:
        void init(double _n, const dVector &y);
        void update(double y_i);
        void reset();
        double node_impurity(const dVector &y);
        double get_score();
        bool should_skip();
    protected:
        double sum_ylogy_l;
        double sum_ylogy_r;
        double sum_ylogy;
    private:
        bool skip = false;
};

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
    return 2*(y.array()*log(y.array()/pred) - (y.array()-pred)).mean();
}

bool Poisson::should_skip(){
    return skip;
}
    
void Poisson::update(double y_i){
    sum_y_l+= y_i;
    sum_y_r-=y_i;
    if(sum_y_l<=eps || sum_y_r<=eps){
        skip = true;
    }
    n_l+=1;
    n_r-=1;
    sum_ylogy_l+= (y_i+eps)*log(y_i+eps);
    sum_ylogy_l-= (y_i+eps)*log(y_i+eps);

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;

    double sum_ylogpred_l = (y_bar_l+eps)*log(y_bar_l+eps)*n_l;
    double sum_ylogpred_r = (y_bar_r+eps)*log(y_bar_r+eps)*n_r;
    score = (sum_ylogy_l-sum_ylogpred_l+ sum_ylogy_r - sum_ylogpred_r)/n;
}

void Poisson::reset(){
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_r = n;
    n_l = 0;
    sum_ylogy_l = 0;
    sum_ylogy_r = sum_ylogy;

}

#endif