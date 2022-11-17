#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#define __SLITTER_HPP_INCLUDED__

//#include <C:\Users\mb-92\OneDrive\Skrivebord\studie\StableTrees\cpp\thirdparty\eigen\Eigen/Dense>
#include <Eigen/Dense>
#include <unordered_set>

#include <omp.h>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;


using namespace std;



class Splitter{

    public:
        
        // double get_split( dMatrix &X, dVector &y);
        double sum_squared_error(const dVector &y_true, double  y_pred);
        tuple<double,double,dVector,dVector > get_predictions(const dVector &feature, const dVector &y, double value);
        double mse_criterion(const dVector  &feature,const dVector  &y, double  value);
        tuple<double,double> select_split(const dVector  &feature, const dVector  &y,const iVector sorted_index);
        tuple<int,double,double>  find_best_split(const dMatrix  &X, const dVector  &y);
        tuple<double,double> select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index);
        iMatrix sorted_indices(dMatrix X);
        vector<int> sort_index(const dVector &v);

    //private:
        //double mse_criterion(dMatrix &feature, dVector &y, bVector &mask);
        //tuple<double,double> select_split(dVector&feature, dVector &y);
        //tuple<int,double,double> select_split(dMatrix&X, dVector &y);

};



tuple<double,double, dVector, dVector> Splitter::get_predictions(const dVector &feature,const  dVector &y,double value){
    double left_prediction = 0.0;
    double right_prediction = 0.0;
    std::vector<double> left_values;
    std::vector<double> right_values;
    for(int i=0; i<y.rows();i++){
        if(feature(i)<=value){
            left_prediction+=y(i);
            left_values.push_back(y(i));
        }else{
            right_prediction+=y(i);
            right_values.push_back(y(i));
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

double Splitter::sum_squared_error(const dVector &y_true, double  y_pred){
    
    return (y_true.array() - y_pred).pow(2.0).sum();

}


double Splitter::mse_criterion(const dVector  &feature, const dVector  &y, double  value){

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
    
    return (right+left)/y.size();

}


tuple<double,double> Splitter::select_split_from_all(const dVector  &feature, const dVector  &y, const iVector sorted_index){
    double min_score = std::numeric_limits<double>::infinity();
    float n = y.size();
    double best_split_value;

    double y_L = 0;
    double y_R = y.array().sum();
    float N_L = 0;
    float N_R = n;
    double y_squared = y.array().square().sum();
    for (int i = 0; i < sorted_index.rows()-1; i++) {
        int low = sorted_index[i];
        int high = sorted_index[i+1];
        double lowValue = feature[low];
        double hightValue = feature[high];


        double split_value =  (lowValue+hightValue)/2;
        y_L+= y(low);
        y_R-= y(low);
        N_L+=1;
        N_R-=1;
        

        double SSE_L= N_L*pow((y_L/N_L),2);
        double SSE_R= N_R*pow((y_R/N_R),2);
        double score = y_squared - SSE_L - SSE_R;

        if(min_score>score){
            min_score = score;
            best_split_value = split_value;
        }

    }
  
    
    return tuple<double,double>(min_score,best_split_value);
}


tuple<double,double> Splitter::select_split(const dVector  &feature,const dVector  &y, const iVector sorted_index){
    // double score;
    // double split_value = feature.array().mean();
    double min_score = std::numeric_limits<double>::infinity();
    
    double best_split_value;
    int n = feature.size();
    int size = min(n,90);
    dVector percentile = dVector::LinSpaced(size, 0.05, 0.95).array();
    int lastInd = 0;
    for (int i = 0; i < percentile.size(); i++) {
        int N = sorted_index.rows();
        int ind = sorted_index(int(N*percentile[i]));
        ind = min(N-2,ind);
        if(lastInd == ind){
            continue;
        }
        lastInd = ind;
        int upper_ind = ind +1;
        double split_value  =  (feature(ind)+feature(upper_ind))/2;
        double score = mse_criterion(feature, y, split_value);
        if(min_score>score){
            min_score = score;
            best_split_value = split_value;
        }
    }
  
    
    return tuple<double,double>(min_score,best_split_value);
}

tuple<int, double,double> Splitter::find_best_split(const dMatrix  &X, const dVector  &y){
    
    
    
        double min_score = std::numeric_limits<double>::infinity();
        double best_split_value;
        int split_feature;
        dVector feature;
        double score;
        double split_value;
        int i;
         /*
        for (int i = 0; i < X.rows(); i++)
            {
            for (int j = 0; j < X.cols(); j++)
            {
                cout << X(i,j) << " ";
            }
                
            // Newline for new row
            cout << endl;
            }

        */

        iMatrix X_sorted_indices = sorted_indices(X);
        /*printf("X_sorted_indices \n");
        for (int i = 0; i < X_sorted_indices.rows(); i++)
            {
            for (int j = 0; j < X_sorted_indices.cols(); j++)
            {
                cout << X_sorted_indices(i,j) << " ";
            }
                
            // Newline for new row
            cout << endl;
            }*/


        #pragma omp parallel for ordered num_threads(4) shared(min_score,best_split_value,split_feature) private(i,score,split_value, feature)
        
        for(int i =0; i<X.cols(); i++){
            feature = X.col(i);
            iVector sorted_index = X_sorted_indices.col(i);
           /* std::unordered_set<double> q{};
            if(i ==0){
                for(int i = 0; i< feature.rows(); i++){
                    q.insert(feature(i));
                }
            }
            printf("%d \n", q.size());*/
            /*for (auto it = q.begin(); it !=q.end(); ++it)
                cout << ' ' << *it;
            cout << endl;*/
            //double split_value;
            //double score;
            tie(score, split_value) = select_split_from_all(feature, y, sorted_index);
            //printf("num obs %d,split_value %f , min_score %f \n", feature.size(),split_value,min_score);
            
            #pragma omp ordered
            {
            if(min_score>score){
                min_score = score;
                best_split_value = split_value;
                split_feature = i;
            }
            }
        }
        //printf("=== %d, %f, %f  \n ",split_feature,min_score, best_split_value);
        return tuple<int, double,double>(split_feature,min_score, best_split_value);
    
    
}

vector<int> Splitter::sort_index(const dVector &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

iMatrix Splitter::sorted_indices(dMatrix X){
    const int nrows = X.rows();
    const int ncols = X.cols();
    iMatrix X_sorted_indices(nrows,ncols);
    
    for(int i = 0; i<ncols; i++){
        vector<int> sorted_ind = sort_index(X.col(i));
        for(int j = 0; j<nrows; j++){
            X_sorted_indices(j,i) = sorted_ind[j];
        }
    }
    return X_sorted_indices;
}


#endif