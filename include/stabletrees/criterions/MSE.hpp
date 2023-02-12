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
        double lamda; //only used in Regulazation criterions
        ~MSE();
    protected:
        double y_sum_squared;
        double optimism_;
};

MSE::~MSE(){
    Criterion::~Criterion();
    y_sum_squared = NULL;
    lamda = NULL;
}

double MSE::get_score(){
    return score;
}

void MSE::init(double _n, const dVector &y){
    
    Criterion::init(_n,y);
    y_sum_squared = y.array().square().sum();
    score = 0;
    node_score = (y_sum_squared - n*(pred*pred))/n;
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
    //printf("MSE::update called \n");
    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    double SSE_L = n_l*( pow(y_bar_l,2.0) );
    double SSE_R = n_r*( pow(y_bar_r,2.0) );
    score = (y_sum_squared - SSE_L-SSE_R)/n;
}

void MSE::reset(){
    Criterion::reset();
}


class MSEABU : public MSE{ 
     public:
        MSEABU(int min_sample_leaf);
        void init(double _n, const dVector &y, dMatrix leaf_info);
        void update(double y_i, const iVector &sorted_index,const dArray &feature_sample, const dMatrix &leaf_info, double split_value);
        
        bool should_skip(int min_sample_leaf);
        void reset();
        ~MSEABU();
    protected:
        double y1_sum_squared;
        double y1_sum_squared_l;
        double y1_sum_squared_r;
        double n1;
        double n1_l;
        double n1_r;
        double sum_y1_l;
        double sum_y1_r;
        double sum_y1;
        double y1_var_l;
        double y1_var_r;
        double y1_var;
        double w_var_l;
        double w_var_r;
        double w_var;
        double gamma;
        double gamma_r;
        double gamma_l;
        int j;
        int min_sample_leaf;
};
bool MSEABU::should_skip(int min_sample_leaf){
    return MSE::should_skip(min_sample_leaf);
}
MSEABU::MSEABU(int min_sample_leaf):MSE(){
    MSE();
    this->min_sample_leaf = min_sample_leaf;
}

MSEABU::~MSEABU(){
    MSE::~MSE();
    y1_sum_squared = NULL;
    y1_sum_squared_l = NULL;
    y1_sum_squared_r = NULL;
    
    n1 = NULL;
    n1_l = NULL;
    n1_r = NULL;

    sum_y1 = NULL;
    sum_y1_l = NULL;
    sum_y1_r = NULL;

    gamma = NULL;
    gamma_r = NULL;
    gamma_l = NULL;
    // y1_var = NULL;
    // y1_var_r = NULL;
    // y1_var_l = NULL;

    // w_var = NULL;
    // w_var_l = NULL;
    // w_var_l = NULL;

    j = NULL;
}
void MSEABU::init(double _n, const dVector &y, dMatrix leaf_info){
    MSE::init(_n,y);
    y1_sum_squared = leaf_info.col(0).array().square().sum();
    y1_sum_squared_l = 0;
    y1_sum_squared_r = y1_sum_squared;
    n1 = leaf_info.rows();
    n1_l = 0;
    n1_r = n1;
    dVector sums =  leaf_info.colwise().sum();
    sum_y1 = sums(0,0);
    sum_y1_l = 0;
    sum_y1_r = sum_y1;
    
    gamma = sums(1,0);

    // y1_var = sums(1,0);
    // y1_var_r = y1_var;
    // y1_var_l = 0;

    // w_var = sums(2,0);
    // w_var_r = w_var;
    // w_var_l = 0;
    j=0;
}

void MSEABU::reset(){
    MSE::reset();
    n1_l = 0;
    n1_r = n1;
    sum_y1_l = 0;
    sum_y1_r = sum_y1;

    // y1_var_r = y1_var;
    // y1_var_l = 0;

    // w_var_r = w_var;
    // w_var_l = 0;
    gamma_l = 0;
    gamma_r = gamma;
    j=0;
    y1_sum_squared_l =0;
    y1_sum_squared_r = y1_sum_squared;
}

void MSEABU::update(double y_i, const iVector &sorted_index,const dArray &feature_sample, const dMatrix &leaf_info, double split_value){

    
    while (true)
    {
        if(j>(n1-2)){
            break;
        }
        
        int low = sorted_index[j];
        double lowValue = feature_sample[low];
        if(lowValue>split_value){
            break;
        }
            
        double y1_i = leaf_info(low,0);
        double gamma_i = leaf_info(low,1);
        //double y1_var_i = leaf_info(low,1);
        //double w_var_i = leaf_info(low,2);

        y1_sum_squared_l += y1_i*y1_i;
        y1_sum_squared_r -= y1_i*y1_i;

        sum_y1_l += y1_i;
        sum_y1_r -= y1_i;
        gamma_l += gamma_i;
        gamma_r -= gamma_i;
        // y1_var_l += y1_var_i;
        // y1_var_r -= y1_var_i;
        
        // w_var_l += w_var_i;
        // w_var_r -= w_var_i;
        n1_l += 1;
        n1_r -= 1;
        j += 1;
        
        
        // if(n1_l < min_sample_leaf || n1_r <min_sample_leaf){
        //     continue;
        // }         
    }
    MSE::update(y_i);
    double y1_bar_l = sum_y1_l/n1_l;
    double y1_bar_r = sum_y1_r/n1_r;
    // double y1_var_bar_l = y1_var_l/n1_l;
    // double y1_var_bar_r = y1_var_r/n1_r;
    // double w_var_bar_l = w_var_l/n1_l;
    // double w_var_bar_r = w_var_r/n1_r;
    double gamma_bar_l = gamma_l/n1_l;
    double gamma_bar_r = gamma_r/n1_r;

    double SSE1_L = n1_l*( pow(y1_bar_l,2.0) );
    double SSE1_R = n1_r*( pow(y1_bar_r,2.0) );
    
    
    
    // double var_ratio_l = y1_var_bar_l/w_var_bar_l/n1_l;
    // //printf("var_ratio_l %f %f %f %f \n", var_ratio_l,y1_var_bar_l,w_var_bar_l, n1_l);
    // double var_ratio_r = y1_var_bar_r/w_var_bar_r/n1_r;

    // std::cout << "w_var_l: " <<  w_var_l<< std::endl;
    // std::cout << "w_var_r: " <<  w_var_r << std::endl;
    // std::cout << "y1_var_l: " <<  y1_var_l<< std::endl;
    // std::cout << "y1_var_r: " <<  y1_var_r << std::endl;
    double reg = (gamma_bar_l*(y1_sum_squared_l - SSE1_L) + gamma_bar_r*(y1_sum_squared_r -SSE1_R))/n1;
    score_reg = score+reg;
    // if(score_reg < std::numeric_limits<double>::infinity()){
    //     std::cout << "score_reg ==inf" << std::endl;
    //     std::cout << "score: " <<  score << std::endl;
    //     std::cout << "score_reg: " <<  score_reg << std::endl;
    //     std::cout << "n1: " <<  n1 << std::endl;
    //     std::cout << "n1_l: " <<  n1_l << std::endl;
    //     std::cout << "n1_r: " <<  n1_r << std::endl;
    //     std::cout << "y1_sum_squared_l: " <<  y1_sum_squared_l << std::endl;
    //     std::cout << "y1_sum_squared_r: " <<  y1_sum_squared_r << std::endl;
    //      std::cout << "SSE1_L: " <<  SSE1_L << std::endl;
    //     std::cout << "SSE1_R: " <<  SSE1_R << std::endl;
    //     std::cout << "gamma_bar_l: " <<  gamma_bar_l<< std::endl;
    //     std::cout << "gamma_bar_r: " <<  gamma_bar_r <<"\n"<< std::endl;
        
        
    //     // std::cout << "y1_var_l: " <<  y1_var_l<< std::endl;
    //     // std::cout << "y1_var_r: " <<  y1_var_r << std::endl;
         
    // }
    //score += reg;
    //printf("update real %f %f %f %f\n ", n, node_score - score, optimism,reg);
}


class MSEReg : public MSE{ 
    public:
        void init(double _n, const dVector &y,const dVector &yprev);
        void update(double y_i, double yp_i);
        void reset();
        ~MSEReg();
    protected:
        double yprev_sum_squared;
        double sum_yprev_l;
        double sum_yprev_r;
        double sum_yprev;
        double node_stability;
};

MSEReg::~MSEReg(){
    MSE::~MSE();
    yprev_sum_squared = NULL;
    sum_yprev_l = NULL;
    sum_yprev_r = NULL;
    sum_yprev = NULL;
    node_stability = NULL;
}

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
    node_stability = (yprev.array() - pred).square().mean();
    //printf("node_score : %f %f \n", node_score, node_score + lambda*node_stability);
    node_score = node_score + lambda*node_stability;

    //printf("G : %f %f \n", G, G + 2*lambda*(pred - yprev.array()).sum());
    G += 2*lambda*(pred - yprev.array()).sum();
    //printf("H : %f %f \n", H, H + 2*lambda*n);
    H += 2*lambda*n;
    
    //printf("G2 : %f %f \n", G2, (2*(pred - y.array()) + 2*lambda*(pred - yprev.array())).square().sum());
    
    G2 = (2*(pred - y.array()) + 2*lambda*(pred - yprev.array()) ).square().sum();
    //printf("H2 : %f %f \n", H2, pow(2 + 2*lambda,2)*n);
    H2 = pow(2 + 2*lambda,2)*n;
    //printf("gxh : %f %f \n", gxh, ( (2*(pred - y.array()) + 2*lambda*(pred - yprev.array()) )*(2+2*lambda)  ).sum());
    gxh = ( (2*(pred - y.array()) + 2*lambda*(pred - yprev.array()) )*(2+2*lambda)  ).sum();
    
    //printf("optimism : %f %f \n", optimism, 2*(G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n));
    optimism = 2*(G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);
    //printf("%f \n", lambda);
}

void MSEReg::update(double y_i, double yp_i){
    MSE::update(y_i);
    sum_yprev_l+= yp_i;
    sum_yprev_r-=yp_i;

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;

    G_l += 2*lambda*(y_bar_l  -  yp_i) ; H_l += 2*lambda;
    double G_r = G - G_l; double H_r = H-H_l;

    double reg_ = (n_l*pow(y_bar_l,2.0) + n_r*pow(y_bar_r,2.0) - 2*sum_yprev_l*y_bar_l - 2*sum_yprev_r*y_bar_r + yprev_sum_squared)/n;
    reg = ( (1+ score) /(1+ node_score) )   +   (lambda) * ((1+reg_)/(1+node_stability));
    //printf("%f %f %f \n",lambda, score/node_score,reg/node_stability);
    score = score  +   lambda * reg_;
    
    
}

#endif