#include <RcppEigen.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

// Enables Eigen
// [[Rcpp::depends(RcppEigen)]]


using namespace Eigen;
using namespace Rcpp;

template <class T>
using Tvec = Eigen::Matrix<T,Dynamic,1>;

template <class T>
using Tmat = Eigen::Matrix<T,Dynamic,Dynamic>;

template<class T>
using Tavec = Eigen::Array<T,Eigen::Dynamic,1>; 



// ------- GUMBEL -------


// Distribution function templated
template<class T>
T pgumbel(double q, T location, T scale, bool lower_tail, bool log_p){
    
    T z = (q-location)/scale;
    T log_px = -exp(-z); // log p(X <= x)
    T res;
    
    if(lower_tail && log_p){
        res = log_px;
    }else if(lower_tail && !log_p){
        res = exp(log_px);
    }else if(!lower_tail && log_p){
        res = log(1.0 - exp(log_px));
    }else{
        res = 1.0 - exp(log_px);
    }
    
    if( std::isnan(res) ){
        return 1.0;
    }else{
        return res;
    }
    
}

// Gradient of estimating equation for scale
double grad_scale_est_obj(double scale, Tavec<double> &x){
    
    int n = x.size();
    Tavec<double> exp_x_beta = (-1.0*x/scale).exp();
    //exp_x_beta = exp_x_beta.array().exp();
    double f = scale + (x*exp_x_beta).sum()/exp_x_beta.sum() - x.sum()/n;
    double grad = 2.0*f* ( 1.0 + 
                           ( (x*x*exp_x_beta).sum() * exp_x_beta.sum() - 
                           pow((x*exp_x_beta).sum(),2.0) ) / 
                           pow(scale*exp_x_beta.sum(), 2.0));
    return grad;
    
}

// ML Estimate of scale
double scale_estimate(Tavec<double> &x){
    
    // Start in variance estimate -- already pretty good
    int n = x.size();
    int mean = x.sum()/n;
    double var = 0.0;
    for(int i=0; i<n; i++){
        var += (x[i]-mean)*(x[i]-mean)/n;
    }
    double scale_est = sqrt(var*6.0)/M_PI;
    
    // do some gradient iterations to obtain ML estimate
    int NITER = 50; // max iterations
    double EPS = 1e-2; // precision
    double step_length = 0.2; //conservative
    double step;
    for(int i=0; i<NITER; i++){
        
        // gradient descent
        step = - step_length * grad_scale_est_obj(scale_est, x);
        scale_est += step;
        
        //Rcpp::Rcout << "iter " << i << ", step: " << std::abs(step) << ", estimate: " <<  scale_est << std::endl;
        
        // check precision
        if(std::abs(step) <= EPS){
            break;
        }
        
    }
    
    return scale_est;
    
}

// ML Estimates
Tvec<double> par_gumbel_estimates(Tavec<double> &x){
    
    int n = x.size();
    
    double scale_est = scale_estimate(x);
    double location_est = scale_est * ( log((double)n) - log( (-1.0*x/scale_est).exp().sum() ) );
    
    Tvec<double> res(2);
    res << location_est, scale_est;
    
    return res;
    
}


// ------- CIR -------


/*
 * cir_sim_vec:
 * Returns cir simulation on transformed equidistant grid tau = f(u)
 */
Tvec<double> cir_sim_vec(int m)
{
    double EPS = 1e-12;
    
    // Find original time of sim: assumption equidistant steps on u\in(0,1)
    double delta_time = 1.0 / ( m+1.0 );
    Tvec<double> u_cirsim = Tvec<double>::LinSpaced(m, delta_time, 1.0-delta_time);
    
    // Transform to CIR time
    Tvec<double> tau = 0.5 * log( (u_cirsim.array()*(1-EPS))/(EPS*(1.0-u_cirsim.array())) );
    
    // Find cir delta
    Tvec<double> tau_delta = tau.tail(m-1) - tau.head(m-1);
    
    // Parameters of CIR
    double a = 2.0;
    double b = 1.0;
    double sigma = 2.0*sqrt(2.0);
    double ncchisq;
    double c = 0.0;
    /*
     double kappa=2.0, sigma = 2.0*sqrt(2.0);
     double a = kappa;
     double b = 2.0 * sigma*sigma / (4.0 * kappa);
     double c = 0;
     double ncchisq;
     */
    
    // Simulate first observation
    // scale = 1/rate = 1/(2a/sigma^2) = sigma^2/(2*a)
    // shape = 2*a*b/sigma^2
    Tvec<double> res(m);
    res[0] = R::rgamma( 0.5, 2.0 );
    
    // Simulate remaining observatins
    for(int i=1; i<m; i++){
        
        c = 2.0 * a / ( sigma*sigma * (1.0 - exp(-a*tau_delta[i-1])) );
        ncchisq =  R::rnchisq( 4.0*a*b/(sigma*sigma), 2.0*c*res[i-1]*exp(-a*tau_delta[i-1]) );
        res[i] = ncchisq/(2.0*c);
        
    }
    
    return res;
    
}

/*
 * cir_sim_mat:
 * Returns 1000 by 1000 cir simulations
 */
// [[Rcpp::export("cir_sim_mat")]]
Tmat<double> cir_sim_mat(int nsim, int nobs)
{
    //int n=100, m=200;
    Tmat<double> res(nsim, nobs);
    
    for(int i=0; i<nsim; i++){
        res.row(i) = cir_sim_vec(nobs);
    }
    
    return res;
    
}


/*
 * interpolate_cir:
 * Returns interpolated observations between an observation vector, u, 
 * and pre-simlated cir observations in cir_sim
 */
Tmat<double> interpolate_cir(const Tvec<double>&u, const Tmat<double>& cir_sim)
{
    // cir long-term mean is 2.0 -- but do not use this! 
    double EPS = 1e-12;
    int cir_obs = cir_sim.cols();
    int n_timesteps = u.size();
    int n_sim = cir_sim.rows();
    int j=0;
    int i=0;
    
    // Find original time of sim: assumption equidistant steps on u\in(0,1)
    double delta_time = 1.0 / ( cir_obs+1.0 );
    Tvec<double> u_cirsim = Tvec<double>::LinSpaced(cir_obs, delta_time, 1.0-delta_time);
    
    // Transform to CIR time
    Tvec<double> tau_sim = 0.5 * log( (u_cirsim.array()*(1-EPS))/(EPS*(1.0-u_cirsim.array())) );
    Tvec<double> tau = 0.5 * log( (u.array()*(1-EPS))/(EPS*(1.0-u.array())) );
    
    // Find indices and weights of for simulations
    Tvec<int> lower_ind(n_timesteps), upper_ind(n_timesteps);
    Tvec<double> lower_weight(n_timesteps), upper_weight(n_timesteps);
    
    // Surpress to lower boundary
    for( ; i<n_timesteps; i++ ){
        if(tau[i] <= tau_sim[0]){
            lower_ind[i] = 0;
            lower_weight[i] = 1.0;
            upper_ind[i] = 0;
            upper_weight[i] = 0.0;
        }else{
            break;
        }
    }
    
    // Search until in-between cir_obs timepoints are found
    for( ; i<n_timesteps; i++ ){
        // If at limit and tau > tau_sim --> at limit --> break
        if( tau_sim[cir_obs-1] < tau[i] ){
            break;
        }
        for( ; j<(cir_obs-1); j++){
            if( tau_sim[j] < tau[i] && tau[i] <= tau_sim[j+1] ){
                lower_ind[i] = j;
                upper_ind[i] = j+1;
                lower_weight[i] = 1.0 - (tau[i]-tau_sim[j]) / (tau_sim[j+1]-tau_sim[j]);
                upper_weight[i] = 1.0 - lower_weight[i];
                break; // stop search
            }
        }
    }
    
    // Surpress to upper boundary
    for( ; i<n_timesteps; i++ ){
        if(tau[i] > tau_sim[cir_obs-1]){
            lower_ind[i] = cir_obs-1;
            upper_ind[i] = 0;
            lower_weight[i] = 1.0;
            upper_weight[i] = 0.0;
        }
    }
    
    // Populate the return matrix
    Tmat<double> cir_interpolated(n_sim, n_timesteps);
    cir_interpolated.setZero();
    
    for(i=0; i<n_sim; i++){
        for(j=0; j<n_timesteps; j++){
            cir_interpolated(i,j) = cir_sim( i, lower_ind[j] ) * lower_weight[j] + 
                cir_sim( i, upper_ind[j] ) * upper_weight[j];
            
        }
    }
    
    return cir_interpolated;
}

/*
 * rmax_cir:
 * Simulates maximum of cir observations on an observation vector u
 */
Tavec<double> rmax_cir(const Tvec<double>& u, const Tmat<double>& cir_sim)
{
    // Simulate maximum of observations on a cir process
    // u: split-points on 0-1
    // cir_sim: matrix of pre-simulated cir-processes on transformed interval
    
    int nsplits = u.size();
    int simsplits = cir_sim.cols();
    int nsims = cir_sim.rows();
    Tvec<double> max_cir_obs(nsims);
    
    if(nsplits < simsplits){
        double EPS = 1e-12;
        //int nsplits = u.size();
        
        // Transform interval: 0.5*log( (b*(1-a)) / (a*(1-b)) )
        Tvec<double> tau = 0.5 * log( (u.array()*(1-EPS))/(EPS*(1.0-u.array())) );
        
        // Interpolate cir-simulations 
        Tmat<double> cir_obs = interpolate_cir(u, cir_sim);
        
        // Calculate row-wise maximum (max of each cir process)
        max_cir_obs = cir_obs.rowwise().maxCoeff();
    }else{
        max_cir_obs = cir_sim.rowwise().maxCoeff();
    }
    
    return max_cir_obs.array();
}

/*
 * estimate_shape_scale: 
 * Estimates shape and scale for a Gamma distribution
 * Note: Approximately!
 */
Tvec<double> estimate_shape_scale(const Tvec<double> &max_cir)
{
    
    // Fast estimation through mean and var
    int n = max_cir.size();
    double mean = max_cir.sum() / n;
    double var = 0;
    for(int i=0; i<n; i++){
        var += (max_cir[i] - mean) * (max_cir[i] - mean) / (n-1);
    }
    double shape = mean * mean / var;
    // Surpress scale to minimum be 1.0? not necessary, bug elsewhere
    //double scale = std::max(1.0, var/mean);
    double scale = var / mean;
    Tvec<double> res(2);
    res << shape, scale;
    return res;
    
}

// Empirical cdf p_n(X\leq x)
double pmax_cir(double x, Tvec<double>& obs){
    
    // Returns proportion of values in obs less or equal to x
    int n = obs.size();
    int sum = 0;
    for(int i=0; i<n; i++){
        if( obs[i] <= x ){
            sum++;
        }
    }
    return (double)sum/n;
}

// Composite simpson's rule -- requires n even (grid.size() odd)
double simpson(Tvec<double>& fval, Tvec<double>& grid){
    
    // fval is f(x) on evenly spaced grid
    
    int n = grid.size() - 1;
    double h = (grid[n] - grid[0]) / n;
    double s = 0;
    
    if(n==2){
        s = fval[0] + 4.0*fval[1] + fval[2];
    }else{
        s = fval[0] + fval[n];
        for(int i=1; i<=(n/2); i++) { s += 4.0*fval[2*i-1]; }
        for(int i=1; i<=((n/2)-1); i++) { s += 2.0*fval[2*i]; }
    }
    s = s*h/3.0;
    return s;
    
}



// ######## ------- NODE ------- #######

// CLASS
class node
{
public:
    
    int split_feature; // j
    int obs_in_node; // |I_t|
    
    double split_value; // s_j
    double node_prediction; // w_t
    double node_tr_loss; // -G_t^2 / (2*H_t)
    double prob_node; // p(q(x)=t)
    double local_optimism; // C(t|q) = E[(g+hw_0)^2]/(n_tE[h])
    double expected_max_S; // E[S_max]
    //double split_point_optimism; // C(\hat{s}) = C(t|q)*p(q(x)=t)*(E[S_max]-1)
    double CRt; // p(q(x)=t) * C(t|q) * E[S_max]
    double p_split_CRt; // p(split(left,right) | q(x)=t) * CRt, p(split(left,right) | q(x)=t) \approx nl/nt for left node
    double g_sum_in_node;
    double h_sum_in_node;
    
    // for approximate bayesian update
    double node_variance; // p(q(x)=t) * C(t|q) * E[S_max] / E[h]
    double response_variance; // Var(y|q(x)=t)
    
    node* left;
    node* right;
    
    //node(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
    //     int obs_in_node, int obs_in_parent, int obs_tot);
    
    void createLeaf(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
                    int obs_in_node, int obs_in_parent, int obs_tot, 
                    double _g_sum_in_node, double _h_sum_in_node,
                    double gradient_variance
                    );
    
    node* getLeft();
    node* getRight();
    
    void split_node(Tvec<double> &g, Tvec<double> &h, Tvec<int> &ind, Tmat<double> &X, Tmat<double> &cir_sim, node* nptr, int n, 
                    double next_tree_score, bool greedy_complexities, double learning_rate,
                    int depth=0, int maxDepth = 1); // take out from node?
    
    bool split_information(const Tvec<double> &g, const Tvec<double> &h, const Tvec<int> &ind, const Tmat<double> &X,
                           const Tmat<double> &cir_sim, const int n);
    
    double expected_reduction(double learning_rate = 1.0);
    
    void reset_node(); // if no-split, reset j, s_j, E[S] and child nodes
    
    void print_child_branches(const std::string& prefix, const node* nptr, bool isLeft);
};

/*
 // Constructor
 node::node(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
 int obs_in_node, int obs_in_parent, int obs_tot)
 {
 this->node_prediction = node_prediction;
 this->node_tr_loss = node_tr_loss;
 this->local_optimism = local_optimism;
 this->prob_node = (double)obs_in_node / obs_tot; // prob_node;
 double prob_split_complement = 1.0 - (double)obs_in_node / obs_in_parent; // if left: p(right, not left), oposite for right
 this->p_split_CRt = prob_split_complement * CRt;
 this->obs_in_node = obs_in_node;
 this->left = NULL;
 this->right = NULL;
 }
 */

// METHODS
void node::createLeaf(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
                      int obs_in_node, int obs_in_parent, int obs_tot,
                      double _g_sum_in_node, double _h_sum_in_node,
                      double gradient_variance
                      )
{
    //node* n = new node;
    this->node_prediction = node_prediction;
    this->node_tr_loss = node_tr_loss;
    this->local_optimism = local_optimism;
    this->prob_node = (double)obs_in_node / obs_tot; // prob_node;
    // NOTE TO BERENT + MORTEN -- THINK ABOUT THE NEXT TWO LINES
    // Does this have to do with CLT and var(mean) = var(y)/n?
    // Consider eq 20 in paper... should this not be opposite? trying opposite
    double prob_split_complement = 1.0 - (double)obs_in_node / obs_in_parent; // if left: p(right, not left), oposite for right
    this->p_split_CRt = CRt * obs_in_node / obs_in_parent;
    this->obs_in_node = obs_in_node;
    this->g_sum_in_node = _g_sum_in_node;
    this->h_sum_in_node = _h_sum_in_node;
    this->left = NULL;
    this->right = NULL;
    
    this->node_variance = this->p_split_CRt / (_h_sum_in_node/obs_in_node);
    this->response_variance = gradient_variance / 4; // this is MSE specific
    
    Rcpp::Rcout << 
        "\n"
        << "CRt: " << CRt << "\n"
        << "_h_sum_in_node: " << _h_sum_in_node << "\n"
        << "obs_in_node: " << obs_in_node << "\n"
                << "p_split_CRt: " << this->p_split_CRt << "\n"
                << "w var: " << this->node_variance << "\n"
                <<"y var: " << this->response_variance << "\n"
        << std::endl;
    
    //return n;
}


node* node::getLeft()
{
    return this->left;
}

node* node::getRight()
{
    return this->right;
}

double node::expected_reduction(double learning_rate)
{
    // Calculate expected reduction on node
    node* left = this->left;
    node* right = this->right;
    
    double loss_parent = this->node_tr_loss;
    double loss_l = left->node_tr_loss;
    double loss_r = right->node_tr_loss;
    
    double R = (loss_parent - loss_l - loss_r);
    double CR = left->p_split_CRt + right->p_split_CRt;
    
    return learning_rate*(2.0-learning_rate)*R-learning_rate*CR;
    
}

void node::reset_node()
{
    
    // Reset node
    this->expected_max_S = 0.0;
    this->split_feature = 0;
    this->split_value = 0.0;
    this->p_split_CRt = 0.0;
    this->CRt = 0.0;
    //this->split_point_optimism = 0.0;
    this->left = NULL;
    this->right = NULL;
    
}


// Algorithm 2 in Appendix C
bool node::split_information(const Tvec<double> &g, const Tvec<double> &h, const Tvec<int> &ind, const Tmat<double> &X,
                             const Tmat<double> &cir_sim, const int n)
{
    // 1. Creates left right node
    // 2. Calculations under null hypothesis
    // 3. Loop over features
    // 3.1 Profiles over all possible splits
    // 3.2 Simultaniously builds observations vectors
    // 3.3.1 Build gumbel (or gamma-one-hot) cdf of max cir for feature j
    // 3.3.2 Update joint cdf of max max cir over all features
    // 4. Estimate E[S]
    // 5. Estimate local optimism and probabilities
    // 6. Update split information in child nodes, importantly p_split_CRt
    // 7. Returns false if no split happened, else true
    
    int split_feature =0, n_indices = ind.size(), n_left = 0, n_right = 0, n_features = X.cols(), n_sim = cir_sim.rows();
    double split_val=0.0, observed_reduction=0.0, split_score=0.0, w_l=0.0, w_r=0.0, tr_loss_l=0.0, tr_loss_r=0.0;
    double Gl_final, Gr_final, Hl_final, Hr_final;
    double left_gradient_variance, right_gradient_variance;
    
    // Return value
    bool any_split = false;
    
    // Iterators
    int j, i;
    
    // Sorting 
    //Tvec<double> vm(n_indices);
    Tvec<size_t> idx(n_indices);
    std::iota(idx.data(), idx.data()+idx.size(), 0);
    
    // Local optimism
    double local_opt_l=0.0, local_opt_r=0.0;
    double Gl, Gl2, Hl, Hl2, gxhl, Gr, Gr2, Hr, Hr2, gxhr;    
    double G=0, H=0, G2=0, H2=0, gxh=0;
    
    // Prepare for CIR
    Tvec<double> u_store(n_indices);
    //double prob_delta = 1.0/n;
    double prob_delta = 1.0/n_indices;
    int num_splits;
    Tavec<double> max_cir(n_sim);
    int grid_size = 101; // should be odd
    double grid_end = 1.5*cir_sim.maxCoeff();
    Tvec<double> grid = Tvec<double>::LinSpaced( grid_size, 0.0, grid_end );
    Tavec<double> gum_cdf_grid(grid_size);
    Tavec<double> gum_cdf_mmcir_grid = Tvec<double>::Ones(grid_size);
    Tvec<double> gum_cdf_mmcir_complement(grid_size);
    
    
    // 1. Create child nodes
    node* left = new node;
    node* right = new node;
    
    // 2. Calculations under null hypothesis
    for(i=0; i<n_indices; i++){
        G += g[ind[i]]; H+=h[ind[i]];
        G2 += g[ind[i]]*g[ind[i]]; H2 += h[ind[i]]*h[ind[i]];
        gxh += g[ind[i]]*h[ind[i]];
    }
    
    // 3. Loop over features
    for(j=0; j<n_features; j++){
        
        // 3.1 Profiles over all possible splits
        Gl = 0.0; Hl=0.0; Gl2=0; Hl2=0, gxhl=0;
        //vm = X.col(j);
        std::sort(idx.data(), idx.data() + idx.size(), [&](int a, int b){return X(ind[a],j) < X(ind[b],j);});
        //idx = sort_indexes(vm);
        
        // 3.2 Simultaniously build observations vectors
        u_store.setZero();
        num_splits = 0;
        
        for(i=0; i<(n_indices-1); i++){
            
            // Left split calculations
            Gl += g[ind[idx[i]]]; Hl+=h[ind[idx[i]]];
            Gl2 += g[ind[idx[i]]]*g[ind[idx[i]]]; Hl2 += h[ind[idx[i]]]*h[ind[idx[i]]];
            gxhl += g[ind[idx[i]]]*h[ind[idx[i]]];
            
            // Right split calculations
            Gr = G - Gl; Hr = H - Hl;
            Gr2 = G2 - Gl2; Hr2 = H2 - Hl2;
            gxhr = gxh - gxhl;
            
            // Is x_i the same as next?
            if(X(ind[idx[i+1]],j) > X(ind[idx[i]],j)){
                //if(vm[idx[i+1]] > vm[idx[i]]){
                
                // Update observation vector
                u_store[num_splits] = (i+1)*prob_delta;
                num_splits++;
                
                // Check for new maximum reduction
                split_score = (Gl*Gl/Hl + Gr*Gr/Hr - G*G/H)/(2.0*n);
                if(observed_reduction < split_score){
                    
                    any_split = true;
                    observed_reduction = split_score; // update
                    
                    // Populate nodes with information
                    split_feature = j;
                    split_val = X(ind[idx[i]],j);
                    w_l = -Gl/Hl;
                    w_r = -Gr/Hr;
                    tr_loss_l = -Gl*Gl / (Hl*2.0*n);
                    tr_loss_r = -Gr*Gr / (Hr*2.0*n);
                    n_left = i+1;
                    n_right = n_indices - (i+1);
                    // Eq. 25 in paper
                    local_opt_l = (Gl2 - 2.0*gxhl*(Gl/Hl) + Gl*Gl*Hl2/(Hl*Hl)) / (Hl*(i+1));
                    local_opt_r = (Gr2 - 2.0*gxhr*(Gr/Hr) + Gr*Gr*Hr2/(Hr*Hr)) / (Hr*(n_indices-(i+1)));
                    // Store more information
                    Gl_final = Gl;
                    Gr_final = Gr;
                    Hl_final = Hl;
                    Hr_final = Hr;
                    
                    left_gradient_variance = Gl2 - 2.0*Gl + Gl*Gl / n_left;
                    right_gradient_variance = Gr2 - 2.0*Gr + Gr*Gr / n_right;
                }
                
            }
            
        }
        
        // 3.3 Estimate empirical cdf for feature j
        if(num_splits > 0){
            // At least one split-point
            
            // Get probabilities
            Tvec<double> u = u_store.head(num_splits);
            //Rcpp::Rcout << "u: \n" <<  u << std::endl; // COMMENT REMOVE
            
            // Get observations of max cir on probability observations
            max_cir = rmax_cir(u, cir_sim); // Input cir_sim!
            
            if(num_splits==1){
                
                // Exactly gamma distrbuted: shape 1, scale 2
                
                // Estimate cdf of max cir for feature j
                for(int k=0; k<grid_size; k++){ 
                    gum_cdf_grid[k] = R::pgamma(grid[k], 0.5, 2.0, 1, 0); // lower tail, not log
                }
                
            }else{
                
                // Asymptotically Gumbel
                
                // Estimate Gumbel parameters
                Tvec<double> par_gumbel = par_gumbel_estimates(max_cir);
                
                // Estimate cdf of max cir for feature j
                for(int k=0; k<grid_size; k++){ 
                    gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
                }
            }
            
            // Update empirical cdf for max max cir
            gum_cdf_mmcir_grid *= gum_cdf_grid;
            
        }
        
    }
    
    if(any_split){
        
        // 4. Estimate E[S]
        gum_cdf_mmcir_complement = Tvec<double>::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
        this->expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
        
        // 5. Update information in parent node -- reset later if no-split
        this->split_feature = split_feature;
        this->split_value = split_val;
        // C(s) = C(w|q)p(q)/2 * (E[S_max]-2)
        this->CRt = (this->prob_node)*(this->local_optimism)*(this->expected_max_S);
        //Rcpp::Rcout << "E[S]: " << this->expected_max_S << "\n" << "CRt: " << this->CRt << std::endl;
        //this->split_point_optimism = (local_opt_l*n_left + local_opt_r*n_right)/(2*n) * (this->expected_max_S - 2.0);
        
        
        // 6. Update split information in child nodes
        left->createLeaf(w_l, tr_loss_l, local_opt_l, this->CRt, 
                         n_left, n_left+n_right, n, Gl_final, Hl_final,
                         left_gradient_variance
                         ); // Update createLeaf()
        right->createLeaf(w_r, tr_loss_r, local_opt_r, this->CRt,
                          n_right, n_left+n_right, n, Gr_final, Hr_final,
                          right_gradient_variance
                          );
        //Rcpp::Rcout << "p_left_CRt: " << left->p_split_CRt << "\n" <<  "p_right_CRt:"  << right->p_split_CRt << std::endl;
        
        // 7. update childs to left right
        this->left = left;
        this->right = right;
        
    }
    
    return any_split;
    
}

void node::split_node(Tvec<double> &g, Tvec<double> &h, Tvec<int> &ind, Tmat<double> &X, Tmat<double> &cir_sim, 
                      node* nptr, int n, 
                      double next_tree_score, bool greedy_complexities, double learning_rate,
                      int depth, int maxDepth)
{
    
    // if flags stop
    if(ind.size()<2){
        return;
    }
    
    /*
     // Check depth
     if(depth>=maxDepth){
     return;
     }
     */
    
    //else check split
    // Calculate split information
    bool any_split = nptr->split_information(g, h, ind, X, cir_sim, n);
    
    // Check if a split happened
    if(!any_split){
        return;
    }
    
    // Comment out of working on depth<maxDepth
    double expected_reduction = nptr->expected_reduction();
    
    // Considering additive effects vs interaction effects trade-off?
    if(!greedy_complexities){
        
        // Don't consider trade-off
        // if expected_reduction < 0 then reset node
        // Force at least one split: kind-of approximately considering trade-off by not scaling with learning_rate
        if(expected_reduction < 0 && depth > 0){
            nptr->reset_node();
            return;
        }
        
    }else{
        
        // Consider trade-off
        
        // depth==0: calculate next_tree_score
        if(depth==0){
            // Quadratic approximation
            // Compare with this root loss = next root loss
            // Can perhaps be done a little better...
            next_tree_score = std::max(0.0, nptr->expected_reduction(1.0));
            //next_tree_score = std::max(0.0, expected_reduction * (1.0 - learning_rate*(2.0-learning_rate)) );
        }
        
        double expected_reduction_normalized = nptr->expected_reduction(1.0) / nptr->prob_node;
        //double expected_reduction_normalized = expected_reduction / (nptr->prob_node);
        
        // Check trade-off
        if(expected_reduction_normalized < next_tree_score && depth > 0){
            nptr->reset_node();
            return;
        }
        
    }
    
    // Tests ok: create new left right indices for partition
    int n_left = nptr->left->obs_in_node;
    int n_right = nptr->right->obs_in_node;
    Tvec<int> ind_left(n_left), ind_right(n_right);
    // Any way to get the idx from split_information?...
    Tvec<size_t> idx(ind.size());
    std::iota(idx.data(), idx.data()+idx.size(), 0);
    std::sort(idx.data(), idx.data() + idx.size(), 
              [&](int a, int b){
                  return X(ind[a],nptr->split_feature) < X(ind[b],nptr->split_feature);
              }
    );
    for(int i=0; i<n_left; i++){
        ind_left[i] = ind[idx[i]];
    }
    for(int i=n_left; i<(n_left+n_right); i++){
        ind_right[i-n_left] = ind[idx[i]];
    }
    
    // Run recursively on left
    split_node(g, h, ind_left, X, cir_sim, nptr->left, n, 
               next_tree_score, greedy_complexities, learning_rate, 
               depth+1, maxDepth);
    
    // Run recursively on right 
    split_node(g, h, ind_right, X, cir_sim, nptr->right, n, 
               next_tree_score, greedy_complexities, learning_rate, 
               depth+1, maxDepth);
    
    /*
     // Tests ok: partition data and split child-nodes
     // Create new g, h, and X partitions
     int n_left = nptr->left->obs_in_node;
     int n_right = nptr->right->obs_in_node;
     Tvec<double> vm = X.col(nptr->split_feature);
     Tvec<size_t> idx = sort_indexes(vm);
     Tvec<double> gl(n_left), hl(n_left), gr(n_right), hr(n_right);
     Tmat<double> xl(n_left, X.cols()), xr(n_right, X.cols());
     
     for(int i=0; i<n_left; i++){
     gl[i] = g[idx[i]];
     hl[i] = h[idx[i]];
     xl.row(i) = X.row(idx[i]); 
     }
     for(int i=n_left; i<(n_left+n_right); i++){
     gr[i-n_left] = g[idx[i]];
     hr[i-n_left] = h[idx[i]];
     xr.row(i-n_left) = X.row(idx[i]); 
     }
     
     // Run recursively on left
     split_node(gl, hl, xl, cir_sim, nptr->left, n, 
     next_tree_score, greedy_complexities, learning_rate, 
     depth+1, maxDepth);
     
     // Run recursively on right 
     split_node(gr, hr, xr, cir_sim, nptr->right, n, 
     next_tree_score, greedy_complexities, learning_rate, 
     depth+1, maxDepth);
     */
    
}

void node::print_child_branches(const std::string& prefix, const node* nptr, bool isLeft){
    
    if(nptr != NULL)
    {
        
        std::cout << prefix;
        
        std::cout << (isLeft ? "├──" : "└──" );
        
        // print the value of the node
        // if leaf, print prediction, else print split info.
        if(nptr->left == NULL){
            // is leaf: node prediction
            std::cout << nptr->node_prediction << std::endl;
        }else{
            // not leaf: split information
            std::cout << "(" << nptr->split_feature << ", " << nptr->split_value << ")" << std::endl;
        }
        
        // enter the next tree level - left and right branch
        print_child_branches( prefix + (isLeft ? "|   " : "    "), nptr->left, true);
        print_child_branches( prefix + (isLeft ? "|   " : "    "), nptr->right, false);
        
    }
    
    
}





// ------- GBTREE -------


class GBTREE
{
    
    //private:
public:
    
    node* root;
    GBTREE* next_tree;
    
    GBTREE();
    
    node* getRoot();
    void train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, Tmat<double> &cir_sim,
               bool greedy_complexities, double learning_rate, int maxDepth=1);
    double predict_obs(Tvec<double> &x);
    Tvec<double> predict_data(Tmat<double> &X);
    double getTreeScore();
    double getConditionalOptimism();
    double getFeatureMapOptimism();
    double getTreeOptimism(); // sum of the conditional and feature map optimism
    int getNumLeaves();
    void print_tree();
    void importance(Tvec<double> &importance_vector, double learning_rate);
    int get_tree_depth();
    double get_tree_max_optimism();
    double get_tree_min_hess_sum();
    
    double prediction_variance_obs(Tvec<double> &x);
    Tvec<double> prediction_variance(Tmat<double> &X);
    double response_variance_obs(Tvec<double> &x);
    Tvec<double> response_variance(Tmat<double> &X);
};


// METHODS


GBTREE::GBTREE(){
    this->root = NULL;
    this->next_tree = NULL;
}

node* GBTREE::getRoot(){
    return this->root;
}


void GBTREE::train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, Tmat<double> &cir_sim,
                   bool greedy_complexities, double learning_rate, int maxDepth)
{
    // Check if root exists 
    // Else create root
    int n = g.size();
    
    if(root == NULL){
        // Calculate information
        double G=0, H=0, G2=0, H2=0, gxh=0;
        for(int i=0; i<n; i++){
            G += g[i]; H+=h[i];
            G2 += g[i]*g[i]; H2 += h[i]*h[i];
            gxh += g[i]*h[i];
        }
        double local_optimism = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);
        
        node* root_ptr = new node;
        root_ptr->createLeaf(-G/H, -G*G/(2*H*n), local_optimism, local_optimism, n, n, n, G, H, G2/n);
        root = root_ptr;
        //root = root->createLeaf(-G/H, -G*G/(2*H*n), local_optimism, local_optimism, n, n, n);
    }
    
    // Root-node indices
    Tvec<int> ind(n);
    std::iota(ind.data(), ind.data()+ind.size(), 0);
    
    root->split_node(g, h, ind, X, cir_sim, root, n, 0.0, greedy_complexities, learning_rate, 0, maxDepth);
    
}

double GBTREE::predict_obs(Tvec<double> &x){
    
    node* current = this->root;
    
    if(current == NULL){
        return 0;
    }
    
    
    while(current != NULL){
        if(current->left == NULL && current ->right == NULL){
            return current->node_prediction;
        }
        else{
            
            if(x[current->split_feature] <= current->split_value){
                current = current->left;
            }else{
                current = current->right;
            }
        }
    }
    return 0;
}

Tvec<double> GBTREE::predict_data(Tmat<double> &X){
    
    int n = X.rows();
    Tvec<double> res(n), x(n);
    
    for(int i=0; i<n; i++){
        x = X.row(i);
        res[i] = predict_obs(x);
    }
    return res;
    
}

Tvec<double> GBTREE::prediction_variance(Tmat<double> &X){
    
    int n = X.rows();
    Tvec<double> res(n), x(n);
    
    for(int i=0; i<n; i++){
        x = X.row(i);
        res[i] = prediction_variance_obs(x);
    }
    return res;
    
}

double GBTREE::prediction_variance_obs(Tvec<double> &x){
    
    node* current = this->root;
    
    if(current == NULL){
        return 0;
    }
    
    
    while(current != NULL){
        if(current->left == NULL && current ->right == NULL){
            return current->node_variance;
        }
        else{
            
            if(x[current->split_feature] <= current->split_value){
                current = current->left;
            }else{
                current = current->right;
            }
        }
    }
    return 0;
}

Tvec<double> GBTREE::response_variance(Tmat<double> &X){
    
    int n = X.rows();
    Tvec<double> res(n), x(n);
    
    for(int i=0; i<n; i++){
        x = X.row(i);
        res[i] = response_variance_obs(x);
    }
    return res;
    
}

double GBTREE::response_variance_obs(Tvec<double> &x){
    
    node* current = this->root;
    
    if(current == NULL){
        return 0;
    }
    
    
    while(current != NULL){
        if(current->left == NULL && current ->right == NULL){
            return current->response_variance;
        }
        else{
            
            if(x[current->split_feature] <= current->split_value){
                current = current->left;
            }else{
                current = current->right;
            }
        }
    }
    return 0;
}

double GBTREE::getTreeScore(){
    // Recurse tree and sum leaf scores
    double treeScore = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) {
            //std::cout <<  current->node_prediction << std::endl; 
            treeScore += current->node_tr_loss;
            current = current->right;
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return treeScore;
}

double GBTREE::getConditionalOptimism(){
    // Recurse tree and sum conditional optimism in leaves
    double conditional_opt_leaves = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return conditional_opt_leaves;
}


double GBTREE::getTreeOptimism(){
    
    // Recurse tree and sum p_split_CRt in leaf-nodes
    double tree_optimism = 0.0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            //conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                tree_optimism += current->CRt; // current->split_point_optimism;
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return tree_optimism;
    
}

void GBTREE::importance(Tvec<double> &importance_vector, double learning_rate){
    
    // Recurse tree and sum importance (reduction in generalization loss)
    int importance_feature = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            //conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                importance_feature = current->split_feature;
                importance_vector[importance_feature] += current->expected_reduction(learning_rate);
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return;
    
}

int GBTREE::getNumLeaves(){
    int numLeaves = 0;
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            numLeaves += 1;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return numLeaves;
}

int max_depth_subtree(node* node)
{
    if (node == NULL)
        return 0;
    else
    {
        /* compute the depth of each subtree */
        int lDepth = max_depth_subtree(node->left);
        int rDepth = max_depth_subtree(node->right);
        
        /* use the larger one */
        if (lDepth > rDepth)
            return(lDepth + 1);
        else return(rDepth + 1);
    }
}

int GBTREE::get_tree_depth(){
    // Compute depth for root-subtree
    return max_depth_subtree(this->root);
}

double GBTREE::get_tree_max_optimism(){
    double max_optimism = 0.0;
    double node_optimism;
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0.0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            //conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                node_optimism = current->CRt; // current->split_point_optimism;
                if(node_optimism > max_optimism){
                    max_optimism = node_optimism;
                }
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            return max_optimism;
}

double GBTREE::get_tree_min_hess_sum(){
    double min_hess_sum_in_node = R_PosInf;
    double node_hess_sum;
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0.0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            // leaf
            node_hess_sum = current->h_sum_in_node; // current->split_point_optimism;
            if(node_hess_sum < min_hess_sum_in_node){
                min_hess_sum_in_node = node_hess_sum;
            }
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                // Internal node
                pre->right = NULL; 
                node_hess_sum = current->h_sum_in_node; // current->split_point_optimism;
                if(node_hess_sum < min_hess_sum_in_node){
                    min_hess_sum_in_node = node_hess_sum;
                }
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            return min_hess_sum_in_node;
}


double tree_expected_test_reduction(GBTREE* tree, double learning_rate){
    // This employs the gb-approximation to the loss
    double train_loss_reduction = tree->getTreeScore();
    double optimism = tree->getTreeOptimism();
    double scaled_expected_test_reduction = 
        (-2.0) * learning_rate*(learning_rate/2.0 - 1.0) * train_loss_reduction + // Scaled observed training loss
        learning_rate * optimism;
    return scaled_expected_test_reduction;
}

void GBTREE::print_tree(){
    // Horizontal printing of the tree
    // Prints ( col_num , split_val ) for nodes not leaves
    // Prints node_prediction for all leaves
    root->print_child_branches("", root, false);
}


// ---------------- EXPOSING CLASSES TO R ----------
RCPP_EXPOSED_CLASS(GBTREE)
    
    RCPP_MODULE(gbtree_module){
        using namespace Rcpp;
        class_<GBTREE>("GBTREE")
            .constructor()
            .method("getRoot", &GBTREE::getRoot)
            .method("train", &GBTREE::train)
            .method("predict_obs", &GBTREE::predict_obs)
            .method("predict_data", &GBTREE::predict_data)
            .method("prediction_variance", &GBTREE::prediction_variance)
            .method("response_variance", &GBTREE::response_variance)
            .method("getTreeScore", &GBTREE::getTreeScore)
            .method("getNumLeaves", &GBTREE::getNumLeaves)
            .method("print_tree", &GBTREE::print_tree)
        ;
    }

