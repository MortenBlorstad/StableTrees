from sklearn.base import BaseEstimator 
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import os

from sklearn.metrics import mean_squared_error

cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys
import numpy as np 
sys.path.append(cur_file_path + '\\..\\..\\cpp\\build\\Release\\')
sys.path.append('../python')
from stable_trees import Node,Tree, StableTree, StableTreeReg,MonteCarloTree,Method2
from abc import ABCMeta
from abc import abstractmethod

class BaseRegressionTree(BaseEstimator, metaclass=ABCMeta):
        
        @abstractmethod
        def __init__(self, max_depth, min_samples_split, random_state) -> None:
            
            if max_depth is None:
                max_depth = 2147483647
            self.max_depth = int(max_depth)

            self.min_samples_split = float(min_samples_split)

            if random_state is None:
                random_state = 0
            self.random_state = int(random_state)
            


        def check_input(self,  X, y):
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
        def update(self,X,y):
            pass

        def fit(self,X, y): 
            X,y = self.check_input(X,y)
            self.tree.learn(X,y)
            self.root = self.tree.get_root()
            return self
        
        def predict(self, X):
            return self.tree.predict(X)

        def plot(self):
            '''
            plots the tree. A visualisation of the tree
            '''
            plt.rcParams["figure.figsize"] = (20,10)
            self.__plot(self.root)
            plt.plot(0, 0, alpha=1) 
            plt.axis("off")
            plt.show()

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
                plt.text(x, y,f"{node.text()}", fontsize=8,ha='center') 
                return
            
            
        
            x_left, y_left = x-off_x,y-off_y
            plt.text(x, y,f"{node.text()}", fontsize=8,ha='center')
            plt.text(x, y-2,f"impurity: {node.get_split_score():.3f}", fontsize=8,ha='center')
            plt.text(x, y-4,f"nsamples: {node.nsamples()}", fontsize=8,ha='center')
            plt.annotate("", xy=(x_left, y_left+4), xytext=(x-2, y-4),
            arrowprops=dict(arrowstyle="->"))

            x_right, y_right = x+off_x,y-off_y
            plt.annotate("", xy=(x_right , y_right+4), xytext=(x+2, y-4),
            arrowprops=dict(arrowstyle="->"))
            self.__plot(node.get_left_node(),x_left, y_left, off_x*0.5)
            self.__plot(node.get_right_node() ,x_right, y_right,off_x*0.5)


            



class BaseLineTree(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None):
        
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = Tree(self.max_depth,self.min_samples_split)
    
    def update(self,X,y):
        return self.fit(X,y)

    
class StableTreeM2(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None):
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = Method2(self.max_depth,self.min_samples_split)
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self



class sklearnBase(DecisionTreeRegressor):

    def __init__(self, *, criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0, min_impurity_split=None, ccp_alpha=0):
        super().__init__(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, ccp_alpha)
        

    def update(self,X,y):
        self.fit(X,y)

class StableTree1(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None, delta = 0.1) -> None:
        
        self.root = None
        self.delta = delta
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = StableTree(self.max_depth,self.min_samples_split)
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y, self.delta)
        self.root = self.tree.get_root()
        return self


class StableTree2(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None) -> None:
        
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = StableTreeReg(self.max_depth,self.min_samples_split)
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y, "method1")
        self.root = self.tree.get_root()
        return self


class StableTree3(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None) -> None:
        
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = StableTreeReg(self.max_depth,self.min_samples_split)
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y, "method2")
        self.root = self.tree.get_root()
        return self

from sklearn.ensemble import RandomForestRegressor
class StableTree4(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None) -> None:
        
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = StableTreeReg(self.max_depth,self.min_samples_split)
        
    def fit(self,X,y):
        X,y = self.check_input(X,y)
        rf = RandomForestRegressor(random_state=0).fit(X,y)
        y_pred1 = rf.predict(X)
        self.tree.learn(X,y)
        y_pred2 = self.predict(X)
        self.tree.update_method3(X,y, y_pred1,y_pred2)
        self.root = self.tree.get_root()
        return self
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        y_pred1 = self.predict(X)
        rf = RandomForestRegressor(random_state=0).fit(X,y)
        y_pred2 = rf.predict(X)
        self.tree.update_method3(X,y, y_pred1,y_pred2)
        self.root = self.tree.get_root()
        return self

class StableTree5(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, ntrees = 10,random_state = 0) -> None:
        
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.ntrees = ntrees
        self.tree = MonteCarloTree(self.max_depth,self.min_samples_split,self.ntrees,random_state)
        
    def fit(self,X,y):
        X,y = self.check_input(X,y)
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        return self
        
    def add_noise(self,X):
        stds = np.std(X, axis=0)
        noise = np.random.normal(0, stds/np.sqrt(X.shape[0]/4), size = X.shape)
        return X+noise
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        np.random.seed(self.random_state)
        inds = np.random.choice(a = np.arange(X.shape[0]), size = 2*X.shape[0], replace = True)
        X_eval = self.add_noise(X[inds,:]); y_eval = y[inds]
        self.tree.update(X,y,X,y)
        self.root = self.tree.get_root()
        return self