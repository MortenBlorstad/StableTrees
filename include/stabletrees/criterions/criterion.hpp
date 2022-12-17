#pragma once
#ifndef __CRITERION_HPP_INCLUDED__

#define __CRITERION_HPP_INCLUDED__



#include <Eigen/Dense>
using namespace std;

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;

class Criterion{
    public:
        explicit Criterion();
        virtual void init(double n, const dVector &y);

        virtual void update(double y_i);
        virtual void reset();

        virtual double node_impurity(const dVector &y);
        virtual bool should_skip();

        double get_score();
    protected:
        double sum_y_l;
        double sum_y_r;
        double sum_y;
        double n;
        double n_l;
        double n_r;
        double node_score;
        double score;
        double eps = 0.000000000000001;





    
};
Criterion::Criterion(){

}
bool Criterion::should_skip(){
    return false;
}

double Criterion::node_impurity(const dVector &y){
    return 0;
}

void Criterion::update(double y_i){
    printf("update empty\n");

}


void Criterion::reset(){

}

double Criterion::get_score(){
    return score;
}
void Criterion::init(double _n, const dVector &y){
    n = _n;
    sum_y = y.array().sum();
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_l = 0;
    n_r = n;
}





#endif