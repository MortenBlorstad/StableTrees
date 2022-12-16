
from _stabletrees import Node, Tree, Method2
from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator 
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np

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

class StableTree(BaseRegressionTree):
    def __init__(self, *, max_depth = None, min_samples_split = 2.0, random_state = None):
        self.root = None
        super().__init__(max_depth, min_samples_split, random_state)
        self.tree = Method2(self.max_depth,self.min_samples_split)
        

    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self