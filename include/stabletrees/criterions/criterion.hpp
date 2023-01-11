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
using dArray = Eigen::Array<double,Eigen::Dynamic,1>;

class Criterion{
    public:
        explicit Criterion();
        virtual void init(double n, const dVector &y,const dVector &yprev);
        virtual void init(double n, const dVector &y);

        virtual void update(double y_i, double yp_i);
        virtual void update(double y_i);
        virtual void reset();

        virtual double node_impurity(const dVector &y);
        virtual bool should_skip(int min_samples_leaf);

        double get_score();
        double node_score;
        double observed_reduction;
        double num_splits = 0;
        double optimism;
        
        
        
    protected:
        double sum_y_l;
        double sum_y_r;
        double sum_y;
        double n;
        double n_l;
        double n_r;
        double G; double G_l;
        double H; double H_l;
        double G2; double H2; double gxh;
        double pred;
        
        
        
        double score;
        double eps = 0.000000000000001;    

};

Criterion::Criterion(){

}

bool Criterion::should_skip(int min_samples_leaf){
    // if(n_l < (double)min_samples_leaf || n_r < (double)min_samples_leaf){
    //     printf("%f %f %d\n", n_l, n_r,min_samples_leaf);
    // }
    return n_l < (double)min_samples_leaf || n_r < (double)min_samples_leaf;
}

double Criterion::node_impurity(const dVector &y){
    return 0;
}

void Criterion::update(double y_i){
    sum_y_l+= y_i;
    sum_y_r-=y_i;
    n_l+=1;
    n_r-=1;
}

void Criterion::update(double y_i, double yp_i){

}


void Criterion::reset(){
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_r = n;
    n_l = 0;
    G_l = 0; H_l = 0; 
    
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
    pred = sum_y/n;
    observed_reduction = 0.0;

}

void Criterion::init(double _n, const dVector &y, const dVector &yprev){

}

#endif