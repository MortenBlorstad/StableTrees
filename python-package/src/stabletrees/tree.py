
from _stabletrees import Node, Tree
from _stabletrees import AbuTree as atree
from _stabletrees import AbuTreeI as atreeI
from _stabletrees import NaiveUpdate as NuTree
from _stabletrees import StabilityRegularization as SrTree
from _stabletrees import TreeReevaluation as TrTree


from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator 
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np

criterions = {"mse":0, "poisson":1}



class BaseRegressionTree(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False, random_state : int = None) -> None:
        criterion = str(criterion).lower()
        if criterion not in criterions.keys():
            raise ValueError("Possible criterions are 'mse' and 'poisson'.")
        self.criterion = criterion

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)

        self.min_samples_split = float(min_samples_split)

        if random_state is None:
            random_state = 0
        self.random_state = int(random_state)

        self.adaptive_complexity = adaptive_complexity
        self.min_samples_leaf = min_samples_leaf


    def check_input(self,  X : np.ndarray ,y : np.ndarray):
        if X.ndim <2:
            X = np.atleast_2d(X)
        if np.issubdtype(X.dtype, np.number):
            X = X.astype("double")
        else:
            raise ValueError("X needs to be numeric")
        
        if y.ndim >1:
            raise ValueError("y needs to be 1-d")

        if np.issubdtype(y.dtype, np.number):
            y = y.astype("double")
        else:
            raise ValueError("y needs to be numeric")
        return X,y

    @abstractmethod
    def update(self,X : np.ndarray ,y : np.ndarray):
        pass

    def fit(self,X : np.ndarray ,y : np.ndarray): 
        X,y = self.check_input(X,y)
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)

    def plot(self):
        '''
        plots the tree. A visualisation of the tree
        '''
        plt.rcParams["figure.figsize"] = (20,10)
        self.__plot(self.root)
        plt.plot(0, 0, alpha=1) 
        plt.axis("off")
        

    def __plot(self,node: Node,x=0,y=-10,off_x = 100000,off_y = 15):
        '''
        a helper method to plot the tree. 
        '''
        

        # No child.
        if node is None:
            return

        if node.is_leaf():
            plt.plot(x+10, y-5, alpha=1) 
            plt.plot(x-10, y-5, alpha=1) 
            plt.text(x, y,f"{node.predict():.2f}", fontsize=8,ha='center') 
            plt.text(x, y-2,f"{node.nsamples():.2f}", fontsize=8,ha='center') 
            return
        
        
    
        x_left, y_left = x-off_x,y-off_y
        plt.text(x, y,f"X_{node.get_split_feature()}<={node.get_split_value():.4f}\n", fontsize=8,ha='center')
        plt.text(x, y-2,f"impurity: {node.get_impurity():.3f}", fontsize=8,ha='center')
        plt.text(x, y-4,f"nsamples: {node.nsamples()}", fontsize=8,ha='center')
        plt.annotate("", xy=(x_left, y_left+4), xytext=(x-2, y-4),
        arrowprops=dict(arrowstyle="->"))

        x_right, y_right = x+off_x,y-off_y
        plt.annotate("", xy=(x_right , y_right+4), xytext=(x+2, y-4),
        arrowprops=dict(arrowstyle="->"))
        self.__plot(node.get_left_node(),x_left, y_left, off_x*0.5)
        self.__plot(node.get_right_node() ,x_right, y_right,off_x*0.5)


class BaseLineTree(BaseRegressionTree):
    """
        Baseline: update method - same as the fit method. 
        Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.
    """

    def __init__(self, *,criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5,
                    adaptive_complexity : bool = False, random_state : int = None) -> None:
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf, adaptive_complexity, random_state)
        self.tree = Tree(criterions[self.criterion], self.max_depth,self.min_samples_split,self.min_samples_leaf, self.adaptive_complexity)
    
    def update(self,X : np.ndarray ,y : np.ndarray):
        return self.fit(X,y)
    
class SklearnTree(DecisionTreeRegressor):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 1, adaptive_complexity : bool = False, random_state = None):
        
        super().__init__(criterion = criterion,max_depth= max_depth, min_samples_split= min_samples_split,min_samples_leaf = min_samples_leaf, random_state = random_state)

    def update(self, X,y):
        self.fit(X,y)
        return self
    

class NaiveUpdate(BaseRegressionTree):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False, random_state = None):
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        self.tree = NuTree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity)
    
    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self 
    

    
class TreeReevaluation(BaseRegressionTree):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                  random_state = None,delta : float=0.1):
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        self.tree = TrTree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity)
        self.delta = delta
    
    def fit(self,X : np.ndarray ,y : np.ndarray): 
        X,y = self.check_input(X,y)
        #y = (y - np.min(y))/(np.max(y)-np.min(y))
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        return self
    
    def update(self, X,y):
        X,y = self.check_input(X,y)
        #y = (y - np.min(y))/(np.max(y)-np.min(y))
        self.tree.update(X,y, self.delta)
        self.root = self.tree.get_root()
        return self  
    
class StabilityRegularization(BaseRegressionTree):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                  random_state = None, lmbda :float= 0.5):
        self.root = None
        self.lmbda = lmbda
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        self.tree = SrTree(self.lmbda, criterions[self.criterion], self.max_depth,self.min_samples_split,self.min_samples_leaf, self.adaptive_complexity)
    
    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self  

class AbuTree(BaseRegressionTree):
    """
   

    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False):
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        self.tree = atree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity)
    
    def predict(self, X):
        return self.tree.predict(X)

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self  

class AbuTreeI(BaseRegressionTree):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False):
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        self.tree = atreeI(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity)
    
    def predict(self, X):
        return self.tree.predict(X)

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self  
    
methods = {"baseline":BaseLineTree, "NU":NaiveUpdate , "TR":TreeReevaluation, "SR":StabilityRegularization, "ABU":AbuTreeI}


class StableTree(BaseRegressionTree):
    def __init__(self, method:str = "baseline", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        