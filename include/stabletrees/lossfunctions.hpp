
#ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
#define __LOSSFUNCTIONS_HPP_INCLUDED__
#include <Eigen/Dense>


using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;

using namespace std;


class LossFunction{

    public:
        LossFunction();
        LossFunction(int _citerion);
        ~LossFunction();
        dVector link_function(dVector y_pred );
        double link_function(double y_pred );
        dVector inverse_link_function(dVector y_pred_transformed );
        double inverse_link_function(double y_pred_transformed );
        dVector loss(dVector y_true,dVector y_pred );
        dVector dloss(dVector y_true,dVector y_pred );
        dVector ddloss(dVector y_true,dVector y_pred );
    protected:
        int citerion;
        
};

LossFunction::LossFunction(){
    this->citerion = 0; //"mse"
}
LossFunction::LossFunction(int citerion){
    this->citerion = citerion; 
}

dVector LossFunction::link_function(dVector y_pred ){
    if(citerion ==1)
         y_pred = y_pred.array().log();

    return y_pred;

}
double LossFunction::link_function(double y_pred ){
    if(citerion ==1)
         y_pred = log(y_pred);

    return y_pred;

}
dVector LossFunction::inverse_link_function(dVector y_pred_transformed ){
    if(citerion ==1)
        y_pred_transformed = y_pred_transformed.array().exp();

    return y_pred_transformed;

}
double LossFunction::inverse_link_function(double y_pred_transformed ){
    if(citerion ==1)
        y_pred_transformed = exp(y_pred_transformed);
    return y_pred_transformed;
}


dVector LossFunction::loss(dVector y_true,dVector y_pred  ){
    if(citerion ==0)
        return (y_true.array() - y_pred.array()).square();

    if(citerion ==1)
        return (y_pred.array().exp() - y_true.array()*y_pred.array());

    throw exception("asdada");
}

dVector LossFunction::dloss(dVector y_true,dVector y_pred  ){
    if(citerion ==0){
        return 2*(y_pred.array()- y_true.array());
    }
        

    if(citerion ==1){
        return y_pred.array().exp() - y_true.array(); //1- y_true.array(); //
    }
        

    throw exception("asdada");
}

dVector LossFunction::ddloss(dVector y_true,dVector y_pred  ){
    if(citerion ==0){
        return dVector::Constant(y_true.size(),0,2.0);
    }

    if(citerion ==1){
        return y_pred.array().exp(); //y_pred.array()/ y_true.array().square() ;
    }
        

    throw exception("asdada");
}

#endif