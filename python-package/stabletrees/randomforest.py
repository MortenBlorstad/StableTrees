from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from stabletrees.tree import Tree, ABUTree
import numpy as np
from joblib import Parallel, delayed
class ABUForest():
    def __init__(self, n_estimators=100,min_samples_leaf=5, random_state = 0):
        self.estimators_ = []
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.n_outputs_ =1
        self.n_estimators = n_estimators

    
    
    def _generate_bootstrap_indices(self, random_state, n_samples):
        """Generate bootstrap indices for one estimator."""
        rng = np.random.default_rng(random_state)
        bootstrap_indices = rng.integers(low = 0, high=n_samples-1, size=n_samples)
        
        return bootstrap_indices
    

    def fit(self, X, y, n_jobs=2):
        """Fit the forest of trees in parallel."""
        self.estimators_ = []
        # Function to fit a single tree
        def fit_single_tree(random_state, X, y):
            tree = ABUTree(random_state=random_state, adaptive_complexity=True, min_samples_leaf=self.min_samples_leaf)
            bootstrap_idx = self._generate_bootstrap_indices(random_state,X.shape[0])
            X_sample, y_sample = X[bootstrap_idx], y[bootstrap_idx]
  
            tree.fit(X_sample, y_sample)

            return tree

        # Parallel fitting of each tree
        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(fit_single_tree)(i, X, y) for i in range(self.n_estimators)
        )

        return self
    
    def update(self, X, y, n_jobs=8):
        """Update the forest of trees in parallel."""
        
        # Function to update a single tree
        def update_single_tree(tree,random_state, X, y):

            bootstrap_idx = self._generate_bootstrap_indices(random_state,X.shape[0])
            X_sample, y_sample = X[bootstrap_idx], y[bootstrap_idx]
            info = self.predict_info(X_sample)
            gammas =info[:,1]
            y_prev = info[:,0]
            tree.update(X_sample, y_sample)
            return tree

        # Parallel updating of each tree
        self.estimators_ = Parallel(n_jobs=n_jobs)(
            delayed(update_single_tree)(tree, i, X, y) for i,tree in enumerate(self.estimators_)
        )

        return self
    

    def predict_info(self, X, n_jobs=8):
        """Make predictions using the forest of trees."""
        
        # Parallel prediction across all trees
        all_predictions_info = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_info_tree)(tree, X) for tree in self.estimators_
        )
        
        # Average predictions from all trees
        avg_info = np.mean(all_predictions_info, axis=0)
        return avg_info

    def _predict_info_tree(self, tree, X):
        """Helper function to predict with a single tree."""
        return tree.predict_info(X)


    def predict(self, X, n_jobs=8):
        """Make predictions using the forest of trees."""
        
        # Parallel prediction across all trees
        all_predictions = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_tree)(tree, X) for tree in self.estimators_
        )

        # Average predictions from all trees
        avg_predictions = np.mean(all_predictions, axis=0)
        return avg_predictions

    def _predict_tree(self, tree, X):
        """Helper function to predict with a single tree."""
        return tree.predict(X)

