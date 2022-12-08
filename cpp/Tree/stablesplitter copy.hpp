#pragma once
#include "splitter.hpp"
#include "node.hpp"
#include <stdexcept>

class StableSplitter: public Splitter{
    public:
        StableSplitter();
        virtual tuple<double,double,double> select_split(const dVector  &feature,const dVector  &y, const iVector sorted_index, const dVector &tree1_predictions);
        virtual tuple<double,double,double> select_split(const dVector  &feature,const dVector  &y, const iVector sorted_index, const dVector &tree1_predictions, const dVector &tree2_predictions);
        double node_stability(double y_node1, double y_node2);
        double node_stability(dVector y_node1, dVector y_node2);
        tuple<double,double> stability_criterion(const dVector  &feature, const dVector  &y, double  value, const dVector &tree1_predictions);

        tuple<double,double> stability_criterion(const dVector  &feature, const dVector  &y, double  value, const dVector &tree1_predictions, const dVector &tree2_predictions);
        virtual tuple<int, double,double> find_best_split(const dMatrix  &X, const dVector  &y,const dVector &tree1_predictions);
        virtual tuple<int, double,double> find_best_split(const dMatrix  &X, const dVector  &y,const dVector &tree1_predictions, const dVector &tree2_predictions);
        double get_stability(const dVector  &feature, const dVector  &y, double  value,const dVector &tree1_predictions);
        double get_stability(const dVector  &feature, const dVector  &y, double  value,const dVector &tree1_predictions, const dVector &tree2_predictions);



};

StableSplitter::StableSplitter():Splitter(){
    Splitter();
}

double StableSplitter::node_stability(double y_node1, double y_node2){
    return (max(y_node1,y_node2)+1)/(min(y_node1,y_node2)+1) -1;
    //return abs(log((y_node1+1e-3)/(y_node2+1e-3)));
}

double StableSplitter::node_stability(dVector y1_pred, dVector y2_pred){
    dVector logs = (y1_pred.array()+1e-3)/(y2_pred.array()+1e-3).log();
    return std::sqrt((logs.array()-logs.array().mean()).square().mean())/(logs.size()-1) ;
}


double StableSplitter::get_stability(const dVector  &feature, const dVector  &y, double  value, const dVector &tree1_predictions){
    double tree1_left_pred; double tree1_right_pred; dVector tree1_left_values; dVector tree1_right_values;
    double left_pred; double right_pred; dVector left_values; dVector right_values;
    tie(left_pred, right_pred, left_values, right_values) = get_predictions(feature, y, value);
    tie(tree1_left_pred, tree1_right_pred, tree1_left_values, tree1_right_values) = get_predictions(feature, tree1_predictions, value);
    


    double left_stability = 0.0;
    double right_stability = 0.0;
    double n = y.size();
    double n_left = left_values.size();
    double n_right = right_values.size();
    if(n_left >0){
        left_stability = node_stability(left_pred,tree1_left_pred)*(n_left/n) ;
    }
    if (n_right>0){
        right_stability = node_stability(right_pred,tree1_right_pred)*(n_right/n);
    }
    return left_stability+right_stability;


}

double StableSplitter::get_stability(const dVector  &feature, const dVector  &y, double  value, const dVector &tree1_predictions, const dVector &tree2_predictions){
    double tree1_left_pred; double tree1_right_pred; dVector tree1_left_values; dVector tree1_right_values;
    double tree2_left_pred; double tree2_right_pred; dVector tree2_left_values; dVector tree2_right_values;
   
    tie(tree1_left_pred, tree1_right_pred, tree1_left_values, tree1_right_values) = get_predictions(feature, tree1_predictions, value);
    tie(tree2_left_pred, tree2_right_pred, tree2_left_values, tree2_right_values) = get_predictions(feature, tree2_predictions, value);
    
    if ( tree1_left_values.size() != tree2_left_values.size() || tree1_right_values.size() != tree2_right_values.size() ) {
        throw std::invalid_argument( "not same lenght" );
    }
    
    double left_stability = 0.0;
    double right_stability = 0.0;
    double n = y.size();
    double n_left = tree1_left_values.size();
    double n_right = tree1_right_values.size();
    if(n_left >0){
        left_stability = node_stability(tree1_left_values,tree2_left_values)*(n_left/n) ;
    }
    if (n_right>0){
        right_stability = node_stability(tree1_right_values,tree2_right_values)*(n_right/n);
    }

    return left_stability+right_stability;
}


tuple<double,double> StableSplitter::stability_criterion(const dVector  &feature, const dVector  &y, double  value, const dVector &tree1_predictions){

    double left_pred; double right_pred; dVector left_values; dVector right_values;
    tie(left_pred, right_pred, left_values, right_values) = get_predictions(feature, y, value);

    double mse_new = Splitter::mse_criterion(feature,y,value);

    double stability = get_stability(feature,y,value,tree1_predictions);
    
    return tuple<double,double>(mse_new + stability*mse_new, mse_new);

}

tuple<double,double> StableSplitter::stability_criterion(const dVector  &feature, const dVector  &y, double  value, const dVector &tree1_predictions, const dVector &tree2_predictions){

    double left_pred; double right_pred; dVector left_values; dVector right_values;
    tie(left_pred, right_pred, left_values, right_values) = get_predictions(feature, y, value);
    
    double mse_new = mse_criterion(feature,y,value);

    double stability = get_stability(feature,y,value,tree1_predictions,tree2_predictions);
    
    return tuple<double,double>(mse_new + stability*mse_new, mse_new);

}

tuple<double,double,double> StableSplitter::select_split(const dVector  &feature,const dVector  &y, const iVector sorted_index, const dVector &tree1_predictions){
    double score;double mse;
    // double split_value = feature.array().mean();
    double min_score = std::numeric_limits<double>::infinity();
    double min_mse;
    double best_split_value;
    vector<double> percentile = { .1, 0.25,0.33, 0.5, 0.67, 0.75, 0.9};

    for (int i = 0; i < percentile.size(); i++) {
        int N = sorted_index.rows();
        int ind = sorted_index(int(N*percentile[i]));
        ind = min(N-2,ind);
        int upper_ind = ind +1;
        double split_value  =  (feature(ind)+feature(upper_ind))/2;
 
        tie(score, mse) = stability_criterion(feature, y,  split_value, tree1_predictions);
        //printf("%f, %d \n", score,N);
        if(min_score>score){
            min_score = score;
            min_mse = mse;
            best_split_value = split_value;
        }
    }

    return tuple<double,double,double>(min_score,best_split_value,min_mse);
}


tuple<double,double,double> StableSplitter::select_split(const dVector  &feature,const dVector  &y, const iVector sorted_index, const dVector &tree1_predictions, const dVector &tree2_predictions){
    double score;double mse;
    // double split_value = feature.array().mean();
    double min_score = std::numeric_limits<double>::infinity();
    double min_mse;
    double best_split_value;
    vector<double> percentile = { .1, 0.25,0.33, 0.5, 0.67, 0.75, 0.9};

    for (int i = 0; i < percentile.size(); i++) {
        int N = sorted_index.rows();
        int ind = sorted_index(int(N*percentile[i]));
        ind = min(N-2,ind);
        int upper_ind = ind +1;
        double split_value  =  (feature(ind)+feature(upper_ind))/2;
 
        tie(score, mse) = stability_criterion(feature, y,  split_value, tree1_predictions, tree2_predictions);
        //printf("%f, %d \n", score,N);
        if(min_score>score){
            min_score = score;
            min_mse = mse;
            best_split_value = split_value;
        }
    }

    return tuple<double,double,double>(min_score,best_split_value,min_mse);
}


tuple<int, double,double> StableSplitter::find_best_split(const dMatrix  &X, const dVector  &y,const dVector &tree1_predictions){

        double min_score = std::numeric_limits<double>::infinity();
        double min_mse;
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        double split_value;
        int i;
       
        iMatrix X_sorted_indices = sorted_indices(X);
        


        #pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature,min_mse) private(i,score,split_value, feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
           
            tie(score, split_value,min_mse) = select_split(feature, y, sorted_index,tree1_predictions);

            
            #pragma omp ordered
            {
            if(min_score>score){
                min_score = score;
                best_split_value = split_value;
                split_feature = i;
            }
            }
        }
        return tuple<int, double,double>(split_feature,min_mse, best_split_value);
    
}

tuple<int, double,double> StableSplitter::find_best_split(const dMatrix  &X, const dVector  &y, const dVector &tree1_predictions, const dVector &tree2_predictions){

        double min_score = std::numeric_limits<double>::infinity();
        double min_mse;
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        double split_value;
        int i;
       
        iMatrix X_sorted_indices = sorted_indices(X);
        


        #pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature,min_mse) private(i,score,split_value, feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
           
            tie(score, split_value,min_mse) = select_split(feature, y, sorted_index,tree1_predictions, tree2_predictions);

            
            #pragma omp ordered
            {
            if(min_score>score){
                min_score = score;
                best_split_value = split_value;
                split_feature = i;
            }
            }
        }
        return tuple<int, double,double>(split_feature,min_mse, best_split_value);
    
}