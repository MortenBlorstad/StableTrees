from matplotlib import pyplot as plt
import os

from sklearn.metrics import mean_squared_error

cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')
from stable_trees import Node


if __name__ == "__main__":
    from tree.RegressionTree import BaseLineTree, StableTree1,sklearnBase, StableTree2, StableTree5
    import time
    from sklearn import datasets
    from sklearn.tree import DecisionTreeRegressor,plot_tree
    import numpy as np  
    X  = np.array([[1,1],[1,2],[2,4],[2,3],[4,5]])
    y = X[:,0]*2 + X[:,1]*0.5 
    
    #tree = BaseLineTree(min_samples_split =2)
    

    # X,y= datasets.make_regression(2000,10, random_state=0)
    start = time.time()
    clf = DecisionTreeRegressor(random_state=0, min_samples_split=5)
    clf = clf.fit(X,y)
    y_pred = clf.predict(X)
    mse = mean_squared_error(y, y_pred)
    end = time.time()   
    
    print(f"sklearn: time {end - start}, mse {mse}" )
    
    # plot_tree(clf)
    # plt.show()
    
    start = time.time()
    tree = StableTree5(min_samples_split =5)
    tree.fit(X,y)
    
    y_pred = tree.predict(X)
    mse = mean_squared_error(y, y_pred)
    end = time.time()
    print(f"my impl: time {end - start}, mse {mse}" )
    #tree.plot()
    


  
  
    # X  = np.array([[1,1],[1,2],[2,4],[2,3],[4,5], [2.5,3.5], [6,10], [4,6] ])
    # y = X[:,0]*2 + X[:,1]*0.5 
    # start = time.time()
    # tree.update(X,y)
    # mse = mean_squared_error(y,tree.predict(X))
    # end = time.time()
    # print(f"my impl: time {end - start}, mse {mse}" )
    # tree.plot()

 

    

    
    
   
            