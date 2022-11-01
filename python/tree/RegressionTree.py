from sklearn.tree import DecisionTreeRegressor 
from matplotlib import pyplot as plt

import os

from sklearn.metrics import mean_squared_error

cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys
import numpy as np 
sys.path.append(cur_file_path + '\\..\\..\\cpp\\build\\Release\\')
sys.path.append('../python')
from stable_trees import Node,Tree, StableTree

class BaseTree(DecisionTreeRegressor):
    def __init__(self):
        self.tree = Tree()
        self.root = None
        super().__init__()


    def fit(self,X, y): 
        X,y = self.check_input(X,y)
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        return self

    def update(self,X,y):
        return self.fit(X,y)

    def predict(self, X):
        return self.tree.predict(X)

    

    def check_input(self,  X, y):
        if X.ndim <2:
            X = X.reshape(X.shape[0],-1)
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


class StableTree(BaseTree):
    def __init__(self) -> None:
        super().__init__()
        
    def update(self, X,y):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self



if __name__ == "__main__":
    import time
    seed = 0
    from sklearn import datasets
    from sklearn.model_selection import train_test_split,KFold
    X,y= datasets.make_regression(2000,10, random_state=0)
    y = y + np.max(y)+100

    kf = KFold(n_splits=X.shape[0],random_state=seed,shuffle=True)
    stability = {"method1": [],"method2": [], "baseline":[]}
    mse = {"method1": [],"method2": [], "baseline":[]}
    models = {"method1":StableTree(), "baseline": BaseTree()}
    epsilon = 1e-3
    iteration = 0
    for train_index, test_index in kf.split(X):
        X_12, y_12 = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=seed)

        
        # initial model 
        
        for name, model in models.items():
            model.fit(X1,y1)
            pred1 = model.predict(X_test)
            model.update(X_12,y_12)
            pred2 = model.predict(X_test)
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(np.log((pred1.item()+epsilon)/(pred2.item()+epsilon)))
       
            if iteration % 100 ==0:
                print(f"{iteration}/2000, {name} - stability: {np.std(stability[name]):.3f}, mse: {np.mean(mse[name]):.3f}")
        
        iteration+=1


    for name in models.keys():
        print("="*40)
        print(f"{name} - stability: {np.std(stability[name]):.3f}, mse: {np.mean(mse[name]):.3f}")
        print("="*40)   

    models["method1"].plot()
    models["baseline"].plot()