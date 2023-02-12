from stabletrees import rnchisq,cir_sim_vec,cir_sim_mat, BaseLineTree
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

cir_sim = cir_sim_mat(100,100)


class ABU(BaseLineTree):
    def __init__(self, *, criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 1, adaptive_complexity: bool = False, random_state: int = None) -> None:
        super().__init__(criterion=criterion, max_depth = max_depth, min_samples_split= min_samples_split, min_samples_leaf=min_samples_leaf, adaptive_complexity= adaptive_complexity,random_state= random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        return super().fit(X, y)
    
    def update_tree(X,y,indicator, depth =0):
        if X.shape[0]<2:
            return
        
        observed_reduction = -np.inf
        any_split = False

        grid_end = 1.5*np.max(cir_sim)
        grid = np.linspace(101, 0.0, grid_end )
        gum_cdf_mmcir_grid = np.ones(100)
        gum_cdf_mmcir_complement =  np.zeros(100)
        pred = y.array().mean()
        n = y.size()
        num_splits = 0
        G = np.sum(1 - y/pred)
        G2 = np.sum((1 - y/pred)**2)
        H = np.sum(y/pred**2)
        H2 = np.sum((y/pred**2)**2)
        for i in range(X.shape[0]):
            nl = 0; nr = n
            Gl = 0; Gl2 = 0; Hl=0; Hl2=0; Gr=G; Gr2 = G2; Hr=H; Hr2 = H2;




    def update(self, X: np.ndarray, y: np.ndarray):
        return super().update(X, y)