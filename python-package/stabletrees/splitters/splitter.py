import numpy as np
from stabletrees.optimism.cir import cir_sim_mat, rmax_cir, simpson
from stabletrees.optimism.gumbel import par_gumbel_estimates, pgumbel
# You need to implement or import the corresponding Python functions for:
# cir_sim_mat, rmax_cir, par_gumbel_estimates, pgumbel, simpson

class Splitter:
    def __init__(self, min_samples_leaf=1, total_obs=None, adaptive_complexity=False, 
                 max_features=None, learning_rate=None):
        self.adaptive_complexity = adaptive_complexity
        self.min_samples_leaf = min_samples_leaf
        self.total_obs = total_obs
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.cir_sim = cir_sim_mat(200, 200, 1)
        self.grid_size = 101
        self.grid = np.linspace(0.0, 1.5 * np.max(self.cir_sim), self.grid_size)
        self.gum_cdf_mmcir_grid = np.ones(self.grid_size)

    # def get_reduction(self, g, h, mask_left):
    #     G = np.sum(g)
    #     H = np.sum(h)
    #     Gl = np.sum(g[mask_left])
    #     Hl = np.sum(h[mask_left])
    #     Gr = G - Gl
    #     Hr = H - Hl
    #     n = len(g)
    #     reduction = ((Gl * Gl) / Hl + (Gr * Gr) / Hr - (G * G) / H) / (2 * n)
    #     return reduction

    def find_best_split(self, X, y, g, h, features_indices):
        n = len(y)
        observed_reduction = -np.inf
        any_split = False

        G = np.sum(g)
        H = np.sum(h)
        G2 = np.sum(g**2)
        H2 = np.sum(h**2)
        gxh = np.sum(g * h)
        expected_max_S = None

        split_feature = None
        split_value = False

        grid_end = 1.5 * np.max(self.cir_sim)
        grid = np.linspace(0.0, grid_end, self.grid_size)
        gum_cdf_mmcir_grid = np.ones(self.grid_size)
        optimism = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n)
        w_var = self.total_obs*(n/self.total_obs)*(optimism/(H))
        y_var =  n * (n/self.total_obs) * self.total_obs * (optimism / H ); 

        for j in features_indices:
            sorted_index  = np.argsort(X[:, j])
            feature = X[:, j]
            num_splits = 0
            u_store = np.zeros(n)
            prob_delta = 1.0 / n
            gum_cdf_grid = np.ones(self.grid_size)

            Gl, Gl2, Hl, Hl2 = 0, 0, 0, 0
            largestValue = feature[sorted_index[n-1]]
            for i in range(n - 1):
                low = sorted_index[i]
                high = sorted_index[i + 1]
                lowValue = feature[low]
                highValue = feature[high]
                middle = (lowValue + highValue) / 2

                g_i = g[low]
                h_i = h[low]
                Gl += g_i
                Hl += h_i
                Gl2 += g_i * g_i
                Hl2 += h_i * h_i
                Gr = G - Gl
                Hr = H - Hl
                Gr2 = G2 - Gl2
                Hr2 = H2 - Hl2
                nl = i + 1
                nr = n - nl
                if lowValue == largestValue:
                    break
                if highValue-lowValue<0.00000000001:
                    continue
            
                if nl< self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue

                

                u_store[num_splits] = nl * prob_delta
                num_splits += 1
                
                score = ((Gl**2) / Hl + (Gr**2) / Hr - (G**2) / H) / (2 * n)
                if observed_reduction < score:
                    any_split = True
                    observed_reduction = score
                    split_value = middle
                    split_feature = j
            if self.adaptive_complexity and num_splits > 0:
                u = u_store[:num_splits]
                max_cir = rmax_cir(u, self.cir_sim)  # Needs definition
                par_gumbel = par_gumbel_estimates(max_cir)  # Needs definition
                for k in range(self.grid_size):
                    gum_cdf_grid[k] = pgumbel(grid[k], par_gumbel[0], par_gumbel[1], True, False)  # Needs definition

            gum_cdf_mmcir_grid *= gum_cdf_grid


        
        if any_split and self.adaptive_complexity:
            gum_cdf_mmcir_complement = np.ones(self.grid_size) - gum_cdf_mmcir_grid
            expected_max_S = simpson(gum_cdf_mmcir_complement, grid)  # Needs definition
            CRt = optimism * (n / self.total_obs) * expected_max_S
            expected_reduction = self.learning_rate * (2.0 - self.learning_rate) * observed_reduction * (n / self.total_obs) - self.learning_rate * CRt

            if any_split and n / self.total_obs != 1.0 and expected_reduction < 0.0:
                any_split = False

        return (any_split, split_feature, split_value,observed_reduction,y_var,w_var,expected_max_S)

