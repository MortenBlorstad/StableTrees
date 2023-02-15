#pragma once
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include "abusplitter.hpp"

class AbuTreeI: public Tree{
    public:
        AbuTreeI(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity);
        AbuTreeI();
        virtual void update(dMatrix &X, dVector &y);
        dMatrix predict_info(dMatrix &X);
        tuple<bool,int,double, double,double,double,double,double>  AbuTreeI::find_update_split(const dMatrix &X,const dVector &y, const dVector &g,const dVector &h,const dVector &weights);
    private:
        dVector predict_info_obs(dVector  &obs);
        tuple<dMatrix,dVector> sample_X_y(const dMatrix &X,const dVector &y, int n1);
        dMatrix sample_X(const dMatrix &X, int n1);
        Node* build_updated_tree(const dMatrix &X, const dVector &y, const dVector &g,const dVector &h, const dVector &weights, int depth);
        AbuSplitter* abu_splitter;
};

AbuTreeI::AbuTreeI():Tree(){
    Tree(); 
}

AbuTreeI::AbuTreeI(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity);
}


dVector AbuTreeI::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(2,1);
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            info(0,1) = node->predict();
            if(_criterion ==1){
                info(1,1) = 1/(node->w_var/node->n_samples);
            }
            else{
                info(1,1) = (node->y_var/node->w_var)/node->n_samples;
            }
                
            //printf("asdas %f %f, %f ,%d \n", info(1,1),node->w_var, node->y_var, node->n_samples);
            return info;
        }else{
            //printf("feature %d, value %f, obs %f \n", node->split_feature, node->split_value,obs(node->split_feature));
            if(obs(node->split_feature) <= node->split_value){
                node = node->left_child;
            }else{
                node = node->right_child;
            }
        }
    }
}
dMatrix AbuTreeI::predict_info(dMatrix &X){
    int n = X.rows();
    dMatrix leaf_info(n,2);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        dVector info =predict_info_obs(obs);
        for (size_t j = 0; j < info.size(); j++)
        {
            leaf_info(i,j) = info(j);
        }
    }
    return leaf_info;
}


void AbuTreeI::update(dMatrix &X, dVector &y){
    abu_splitter = new AbuSplitter(min_samples_leaf,total_obs,_criterion, adaptive_complexity);
    dMatrix Xb = sample_X(X,n1);
    dMatrix info = predict_info(Xb);
    dVector weightsb = info.col(1);
    dVector yb = info.col(0);
    
    dVector hb = 2*weightsb.array();

    dVector gb = -1*hb.array().cwiseProduct(yb.array());

    dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1)); 
    dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1) );
    dMatrix X_concat(X.rows()+Xb.rows(), X.cols());
    dVector y_concat(y.rows()+yb.rows(), 1);
    dVector g_concat(g.rows() + gb.rows(), 1); 
    dVector h_concat(h.rows() + hb.rows(), 1); 
    dVector weights(h.rows() + hb.rows(),1);
    dVector non_b_weights;
    if(_criterion == 0){
        non_b_weights = dVector::Ones(h.rows(),1);
    }else if(_criterion == 1){
        non_b_weights = dVector::Zero(h.rows(),1);
    }
    
    
    
    g_concat <<g,gb ;
    h_concat <<h,hb;
    X_concat <<X,Xb;
    y_concat <<y,yb;
    weights << non_b_weights,weightsb;

    // for(int r=0; r<weights.rows(); r++){
    //     cout<<" "<<weights(r)<<"\n";
    // }
    // printf("%f\n",y_concat.array().sum());
    // printf("%f\n",weights.array().sum());
    total_obs = X_concat.rows();
    this->root = build_updated_tree(X_concat, y_concat, g_concat, h_concat,weights, 0);
    n1 = total_obs;
}

tuple<bool,int,double, double,double,double,double,double>  AbuTreeI::find_update_split(const dMatrix &X,const dVector &y, const dVector &g,const dVector &h,const dVector &weights){
    return abu_splitter->find_best_split(X, y, g, h, weights);
}


Node* AbuTreeI::build_updated_tree(const dMatrix &X, const dVector &y, const dVector &g,const dVector &h, const dVector &weights, int depth){
     number_of_nodes +=1;

    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        
        return NULL;
    }

   

    
    int n = y.size();
    double pred = y.array().mean();


    if(all_same(y)){
        return new Node(pred, n,1,1);
    }
    
    bool any_split;
    double score;
    double impurity;
    double split_value;
    double w_var;
    double y_var;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    tie(any_split, split_feature, split_value,impurity, score, y_var ,w_var,expected_max_S)  = find_update_split(X,y, g,h,weights);
    
    
    if(depth>=this->max_depth){
        return new Node(pred, n, y_var,w_var);
    }
    if(y.rows()< this->min_split_sample){
        return new Node(pred, n, y_var,w_var);
    }
    if(!any_split){
         return new Node(pred ,n, y_var, w_var  );
    }

    if(score == std::numeric_limits<double>::infinity()){
        cout<<"\n Two Dimensional Array is : \n";
        for(int r=0; r<X.rows(); r++)
        {
                for(int c=0; c<X.cols(); c++)
                {
                        cout<<" "<<X(r,c)<<" ";
                }
                cout<<"\n";
        }
         cout<<"\n one Dimensional Array is : \n";
        for(int c=0; c<y.size(); c++)
        {
                cout<<" "<<y(c)<<" ";
        }
        cout<<"\n";
    }
   

    dVector feature = X.col(split_feature);

    tie(mask_left, mask_right) = get_masks(feature, split_value);

    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , pred, y_var, w_var);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
    dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);
    dVector weights_left = weights(mask_left,1); dVector weights_right = weights(mask_right,1);
    
    node->left_child = build_updated_tree( X_left, y_left, g_left,h_left,weights_left, depth+1);
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
    }

    node->right_child = build_updated_tree(X_right, y_right,g_right,h_right,weights_right, depth+1) ;
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
    }

    return node;
}



tuple<dMatrix,dVector> AbuTreeI::sample_X_y(const dMatrix &X,const dVector &y, int n1){
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t>  distr(0, X.rows());
    dMatrix X_sample(n1, X.cols());
    dVector y_sample(n1,1);
    for (size_t i = 0; i < n1; i++)
    {
        size_t ind = distr(gen);
        for(size_t j =0; j<= X.cols(); j++){
            X_sample(i,j) = X(ind,j);
        }
        y_sample(i,0) = y(ind,0);
    }   
    return tuple<dMatrix,dVector>(X_sample,y_sample);
}

dMatrix AbuTreeI::sample_X(const dMatrix &X, int n1){
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t>  distr(0, X.rows());
    dMatrix X_sample(n1, X.cols());
    for (size_t i = 0; i < n1; i++)
    {   
        size_t ind = distr(gen);
        for (size_t j = 0; j < X.cols(); j++)
        {
            X_sample(i,j) = X(ind,j);
        } 
    }
    
    return X_sample;
}










