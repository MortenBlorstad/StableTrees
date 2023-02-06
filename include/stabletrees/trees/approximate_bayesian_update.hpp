#pragma once
#include "tree.hpp"
#include "splitterabu.hpp"
class ABU: public Tree{
    public:
        ABU(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf);
        ABU();
        virtual void update(dMatrix &X, dVector &y);
        dMatrix predict_info(dMatrix &X);
        virtual Node* build_tree(dMatrix  &X, dVector &y, int depth);
        void learn(dMatrix  &X, dVector &y);
    private:
        Node* update_tree(dMatrix  &X, dVector &y, int depth,dMatrix &X_sample, dMatrix &leaf_info);
        SplitterABU* update_splitter;
        SplitterABU* splitter;
        dVector predict_info_obs(dVector  &obs);
        dMatrix sample_X(const dMatrix &X, int n1);
        int n1;

};

ABU::ABU():Tree(){
    Tree();
    
}

ABU::ABU(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, true);
}


Node* ABU::build_tree(dMatrix  &X, dVector &y, int depth){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<1 || y.rows()<1){
            return NULL;
    }
    bool any_split;
    double score;
    double impurity;
    double split_value;
    double y_var;
    double w_var;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
  
    tie(any_split, split_feature,impurity, score, split_value,y_var,w_var) = splitter->find_best_split(X, y, depth);
    if(!any_split){
         return new Node(y.array().mean() ,y.rows(),y_var, w_var  );
    }
  
    if(score == std::numeric_limits<double>::infinity()){
        cout << "error learn: "<< std::endl;
        cout << "X len: "<< X.rows()<< std::endl;
        cout << "any_split: "<< any_split<< std::endl;
        cout << "split_feature: "<< split_feature<< std::endl;
        cout << "impurity: "<< impurity<< std::endl;
        cout << "score: "<< score<< std::endl;
        cout << "y_var: "<< split_value<< std::endl;
        cout << "y_var: "<< y_var<< std::endl;
        cout << "w_var: "<< w_var<<"\n" << std::endl;

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
    double pred = y.array().mean();
    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , pred, y_var, w_var);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    node->left_child = build_tree( X_left, y_left, depth+1);
   
    
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    node->right_child = build_tree(X_right, y_right,depth+1) ;

    return node;
    
}



dVector ABU::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(2,1);
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            info(0,1) = node->predict();
            info(1,1) = node->y_var/node->w_var/node->n_samples;
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
dMatrix ABU::predict_info(dMatrix &X){
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

void ABU::learn(dMatrix  &X, dVector &y){
    total_obs = y.size();
    n1 = total_obs;
    splitter = new SplitterABU(max_depth,min_split_sample, min_samples_leaf,total_obs,_criterion);
    this->root = build_tree(X, y, 0);

}

dMatrix ABU::sample_X(const dMatrix &X, int n1){
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t>  distr(0, X.rows());
    dMatrix X_sample(n1, X.cols());
    for (size_t i = 0; i < n1; i++)
    {
       X_sample(i) = X(distr(gen));
    }
    
    return X_sample;
}

void ABU::update(dMatrix &X, dVector &y) {
    if(this->root == NULL){
        this->learn(X,y);
    }else{
        total_obs = y.size();
        dMatrix X_sample = sample_X(X,n1);
        dMatrix leaf_info = this->predict_info(X_sample);
        update_splitter = new SplitterABU(max_depth,min_split_sample, min_samples_leaf,total_obs,_criterion);
        this->root = update_tree(X,y,0,X_sample, leaf_info);
    } 
}

Node* ABU::update_tree(dMatrix  &X, dVector &y, int depth,dMatrix &X_sample, dMatrix &leaf_info){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<1 || y.rows()<1){
            return NULL;
    }
    bool any_split;
    double score;
    double impurity;
    double split_value;
    double y_var;
    double w_var;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    iVector mask_sample_left;
    iVector mask_sample_right;
    //printf("before\n");
    tie(any_split, split_feature,impurity, score, split_value,y_var,w_var) = update_splitter->find_best_split(X, y,X_sample,leaf_info, depth);
    // printf("after\n");
    //     cout << "any_split: "<< any_split<< std::endl;
    //     cout << "split_feature: "<< split_feature<< std::endl;
    //     cout << "impurity: "<< impurity<< std::endl;
    //     cout << "score: "<< score<< std::endl;
    //     cout << "y_var: "<< split_value<< std::endl;
    //     cout << "y_var: "<< y_var<< std::endl;
    //     cout << "w_var: "<< w_var<<"\n" << std::endl;
    if(!any_split){
         return new Node(y.array().mean() ,y.rows(),y_var, w_var  );
    }
    if(score == std::numeric_limits<double>::infinity()){
        cout << "error update: "<< std::endl;
        cout << "X len: "<< X.rows()<< std::endl;
        cout << "X_sample len: "<< X_sample.rows()<< std::endl;
        cout << "any_split: "<< any_split<< std::endl;
        cout << "split_feature: "<< split_feature<< std::endl;
        cout << "impurity: "<< impurity<< std::endl;
        cout << "score: "<< score<< std::endl;
        cout << "y_var: "<< split_value<< std::endl;
        cout << "y_var: "<< y_var<< std::endl;
        cout << "w_var: "<< w_var<<"\n" << std::endl;
        // cout<<"\n Two Dimensional Array is : \n";
        // for(int r=0; r<X.rows(); r++)
        // {
        //         for(int c=0; c<X.cols(); c++)
        //         {
        //                 cout<<" "<<X(r,c)<<" ";
        //         }
        //         cout<<"\n";
        // }
        //  cout<<"\n one Dimensional Array is : \n";
        
        // for(int c=0; c<y.size(); c++)
        // {
        //         cout<<" "<<y(c)<<" ";
        // }
        // cout<<"\n";
        
    }
    dVector feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, split_value);
    double pred = y.array().mean();
    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , pred, y_var, w_var);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    iVector keep_cols_info = iVector::LinSpaced(leaf_info.cols(), 0, leaf_info.cols()-1).array();

    
    dVector feature_sample = X_sample.col(split_feature);
    tie(mask_sample_left, mask_sample_right) = get_masks(feature_sample, split_value);

    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dMatrix X_sample_left = X(mask_sample_left,keep_cols); dMatrix leaf_info_left = leaf_info(mask_sample_left,keep_cols_info);
    node->left_child = update_tree( X_left, y_left, depth+1,X_sample_left,leaf_info_left);
   
    
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dMatrix X_sample_right = X(mask_sample_right,keep_cols); dMatrix leaf_info_right = leaf_info(mask_sample_right,keep_cols_info);
    node->right_child = update_tree(X_right, y_right, depth+1,X_sample_right,leaf_info_right) ;

    return node;
}







