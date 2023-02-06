#pragma once
#include "tree.hpp"
#include "splitterabu.hpp"
class AbuTree: public Tree{
    public:
        AbuTree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf);
        AbuTree();
        virtual void update(dMatrix &X, dVector &y);
        dMatrix predict_info(dMatrix &X);
        virtual Node* build_tree(dMatrix  &X, dVector &y,dVector g, dVector h, int depth);
        void learn(dMatrix  &X, dVector &y);
    private:
        Node* update_tree(dMatrix  &X, dVector &y, int depth,dMatrix &X_sample, dMatrix &leaf_info);
        dVector predict_info_obs(dVector  &obs);
        tuple<dMatrix,dVector> sample_X_y(const dMatrix &X,const dVector &y, int n1);
        dMatrix sample_X(const dMatrix &X, int n1);
        int n1;
        
        dMatrix cir_sim;
        iMatrix sorted_indices(dMatrix X);
        vector<int> sort_index(const dVector &v);
        int grid_size = 101;
        dVector grid;
        dArray gum_cdf_mmcir_grid;

};

AbuTree::AbuTree():Tree(){
    
    Tree();
    
}

AbuTree::AbuTree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, true);
}


Node* AbuTree::build_tree(dMatrix  &X, dVector &y,dVector g, dVector h, int depth){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        
        return NULL;
    }
    if(depth>max_depth){
        
        return NULL;
    }
    
    
    
    
    
    double score;
    double impurity;
    double split_value;
    double y_var;
    double w_var;
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    

    double observed_reduction = -std::numeric_limits<double>::infinity();
    dVector feature;
    bool any_split = false;
    grid = dVector::LinSpaced(grid_size, 0.0, grid_end );
    gum_cdf_mmcir_grid = dArray::Ones(grid_size);
    int i;
    int n = y.size();
    iMatrix X_sorted_indices = sorted_indices(X);

    
    
    
    double G=g.array().sum(), H=h.array().sum(), G2=g.array().square().sum(), H2=h.array().square().sum(), gxh=(g.array()*h.array()).sum();
    double Gl_final; double Hl_final;
    double grid_end = 1.5*cir_sim.maxCoeff();
    dVector grid = dVector::LinSpaced( grid_size, 0.0, grid_end );
    gum_cdf_mmcir_grid = dVector::Ones(grid_size);
    dVector gum_cdf_mmcir_complement(grid_size);
    double pred = -G/H;
    //printf("pred %f\n", pred);
    int num_splits;
    dVector u_store((int)n);
    double prob_delta = 1.0/n;
    dArray gum_cdf_grid(grid_size);
    double optimism = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);
    w_var = total_obs*(n/total_obs)*(optimism/(H));
    y_var =  n * (n/total_obs) * total_obs * (optimism / H ); //(y.array() - y.array().mean()).square().mean();
    if(all_same(y)){
        return new Node(pred, n,1,1);
    }
    double expected_max_S;

    
    for(int i = 0; i<X.cols(); i++){
        int nl = 0; int nr = n;
        double Gl = 0, Gl2 = 0, Hl=0, Hl2=0, Gr=G, Gr2 = G2, Hr=H, Hr2 = H2;
        feature = X.col(i);
        num_splits = 0;
        iVector sorted_index = X_sorted_indices.col(i);
        double largestValue = feature(sorted_index[n-1]);
        u_store = dVector::Zero(n);
        
        for (int j = 0; j < n-1; j++) {
            int low = sorted_index[j];
            int high = sorted_index[j+1];
            double lowValue = feature[low];
            double hightValue = feature[high];
            double middle =  (lowValue+hightValue)/2;

            double g_i = g(low);
            double h_i = h(low);
            Gl += g_i; Hl += h_i;
            Gl2 += g_i*g_i; Hl2 += h_i*h_i;
        

            Gr = G - Gl; Hr = H - Hl;
            Gr2 = G2 - Gl2; Hr2 = H2 - Hl2;
            nl+=1;
            nr-=1;
            
            // break if rest of the values are equal
            if(lowValue == largestValue){
                break;
            }
            // skip if values are approx equal
            if(hightValue-lowValue<0.00000000001){
                continue;
            }
            if(nl< min_samples_leaf || nr < min_samples_leaf){
                continue;
            }
            
            u_store[num_splits] = nl*prob_delta;
            num_splits +=1;
            score  = ((Gl*Gl)/Hl + (Gr*Gr)/Hr - (G*G)/H)/(2*n);
            any_split = true;
            // printf("%f, %f ,%f,%f, %f ,%f, %d \n",Gl,Gr,G, Hl, Hr, H, n);
            // printf("%d, %d, %d, %f, %f\n",num_splits, nl,nr, score, optimism);
            if(any_split && observed_reduction<score){
                observed_reduction = score;
                split_value = middle;
                split_feature = i;  
                Gl_final = Gl;
                Hl_final = Hl;

            }
        }
        

        
        if(num_splits<=0){
            any_split = false;
        }else{
            dVector u = u_store.head(num_splits);
            dArray max_cir = rmax_cir(u, cir_sim); // Input cir_sim!
            if(num_splits>1){
                // Asymptotically Gumbel
                    
                // Estimate Gumbel parameters
                dVector par_gumbel = par_gumbel_estimates(max_cir);
                // Estimate cdf of max cir for feature j
                for(int k=0; k< grid_size; k++){ 
                    gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
                }

            }else{
                

                // Asymptotically Gumbel
                    
                // Estimate Gumbel parameters
                dVector par_gumbel = par_gumbel_estimates(max_cir);
                // Estimate cdf of max cir for feature j
                for(int k=0; k< grid_size; k++){ 
                    gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
                }
            }
            // Update empirical cdf for max max cir
            gum_cdf_mmcir_grid *= gum_cdf_grid; 
        }
    }
    //printf("any_split %d\n", any_split);
    if(any_split){
        gum_cdf_mmcir_complement = dVector::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
        expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
        double CRt = optimism * (n/total_obs)  *expected_max_S;
        double expected_reduction = 1.0*(2.0-1.0)*observed_reduction*((n/total_obs) )  - 1.0*CRt;
        // std::cout << "local_optimism: " <<  optimism<< std::endl;
        // std::cout << "CRt: " <<  CRt << std::endl;
        // std::cout << "n:  " <<  n  <<std::endl;
        // std::cout << "prob_node:  " <<  n/total_obs << std::endl;
        // std::cout << "expected_max_S:  " <<  expected_max_S << std::endl;
        // std::cout << "observed_reduction:  " <<  observed_reduction << std::endl;
        // std::cout << "expected_reduction:  " <<  expected_reduction <<std::endl;
        // std::cout << "G:  " << G << std::endl;
        // std::cout << "H:  " << H << std::endl;
        // std::cout << "Gl: " <<  Gl_final<< std::endl;
        // std::cout << "Hl: " <<  Hl_final << std::endl;
        // std::cout << "y_var:  " <<  y_var <<std::endl;
        // std::cout << "w_var:  " <<  w_var <<"\n"<<std::endl;

        if(any_split && n/total_obs!=1.0 && expected_reduction<0.0){
            any_split = false;
        }
    }
   
  

    if(!any_split){
        //printf("make leaf  %f %f\n", y_var, w_var);
         return new Node(pred ,n, y_var, w_var  );
    }
    //printf("make internal \n");
    feature = X.col(split_feature);
    tie(mask_left, mask_right) = get_masks(feature, split_value);
    impurity = observed_reduction;
    Node* node = new Node(split_value, impurity, score, split_feature, y.rows() , pred, y_var, w_var);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);
    node->left_child = build_tree( X_left, y_left,g_left, h_left, depth+1);
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
    }

    node->right_child = build_tree(X_right, y_right,g_right,h_right,depth+1) ;
    if(node->right_child!=NULL){
        node->right_child->w_var*=expected_max_S;
    }
    

    return node;
    
}



dVector AbuTree::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(2,1);
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            info(0,1) = node->predict();
            info(1,1) = (node->y_var/node->w_var)/node->n_samples;
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
dMatrix AbuTree::predict_info(dMatrix &X){
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

void AbuTree::learn(dMatrix  &X, dVector &y){
    set_seed(1);
    cir_sim = cir_sim_mat(100,100);
    total_obs = y.size();
    n1 = total_obs;
    dVector g = 2*(- y.array());
    dVector h = dVector::Constant(n1,0,2.0);
    this->root = build_tree(X, y, g, h, 0);
}

void AbuTree::update(dMatrix &X, dVector &y){
    set_seed(1);
    
    cir_sim = cir_sim_mat(100,100);
    dMatrix Xb = sample_X(X,n1);
    dMatrix info = predict_info(Xb);
    dVector weights = info.col(1);
    dVector yb = info.col(0);

    dVector hb = 2*weights.array();

    dVector gb = -1*hb.array().cwiseProduct(yb.array());

    dVector g = 2*(- y.array());
    dVector h = dVector::Constant(X.rows(),0,2.0);
    dMatrix X_concat(X.rows()+Xb.rows(), X.cols());
    dVector y_concat(y.rows()+yb.rows(), 1);
    dVector g_concat(g.rows() + gb.rows(), 1); 
    dVector h_concat(h.rows() + hb.rows(), 1); 
    
    
    g_concat <<g,gb ;
    h_concat <<h,hb;
    X_concat <<X,Xb;
    y_concat <<y,yb;
    //printf("%f %f \n ", hb.array().sum(), h.array().sum());
    // printf("%f\n", gb.array().mean());
    // for(int r=0; r<gb.rows(); r++)
    //     {
    //             for(int c=0; c<gb.cols(); c++)
    //             {
    //                     cout<<" "<<gb(r,c)<<" ";
    //             }
    //             cout<<"\n";
    //     }
    // for (size_t i = 0; i < X_concat.rows(); i++)
    // {
    //     for (size_t j = 0; j < X_concat.cols(); j++)
    //     {
    //         double control;
    //         if(i<X.rows()-1){
    //             control = X(i,j);
    //         }else{
    //             control = Xb(i-X.rows() ,j);
    //         }
    //         printf("(%f, %f)",X_concat(i,j),control);
    //     }
    //     printf("\n");
        
    // }
    total_obs = X_concat.rows();
    this->root = build_tree(X_concat, y_concat, g_concat, h_concat, 0);
    n1 = total_obs;
}

tuple<dMatrix,dVector> AbuTree::sample_X_y(const dMatrix &X,const dVector &y, int n1){
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

dMatrix AbuTree::sample_X(const dMatrix &X, int n1){
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

vector<int> AbuTree::sort_index(const dVector &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] <= v[i2];});

  return idx;
}

iMatrix AbuTree::sorted_indices(dMatrix X){
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








