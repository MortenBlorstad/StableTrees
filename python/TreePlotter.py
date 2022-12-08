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
    from sklearn.datasets import make_regression,load_diabetes
    from sklearn.tree import DecisionTreeRegressor,plot_tree
    import numpy as np  
    X  = np.array([[1,1],[1,2],[2,4],[2,3],[4,5],[6,10], [4,6]])
    y = X[:,0]*2 + X[:,1]*0.5 
    
    #tree = BaseLineTree(min_samples_split =2)
    from sklearn.model_selection import train_test_split,KFold
    # N = 500
    # X,y= make_regression(N, 10, random_state=10)
    # y = y + np.max(y)+100
    #X,y = load_diabetes(return_X_y=True)
    #print(X.shape)
    X1,X2,y1,y2 =train_test_split(X,y,test_size=0.2,random_state=0)
    start = time.time()
    tree = DecisionTreeRegressor(random_state=0).fit(X,y)
    
    mse = mean_squared_error(y1,tree.predict(X1))
    end = time.time()
    print(f"sklearn: time {end - start:.5f}, mse {mse:.5f}" )
    # plot_tree(tree)
  
  
    start = time.time()

    tree = BaseLineTree(min_samples_split = 2, random_state=0).fit(X,y)
    print("sdasd")
    mse = mean_squared_error(y1,tree.predict(X1))
    end = time.time()
    
    print(f"my impl: time {end - start:.5f}, mse {mse:.5f}" )
    start = time.time()
    
    tree = sklearnBase(min_samples_split =2,random_state=0).fit(X,y)
    
    mse = mean_squared_error(y1,tree.predict(X1))
    end = time.time()
    print(f"sklearnBase: time {end - start:.5f}, mse {mse:.5f}" )
    

    # kf = KFold(n_splits=X.shape[0], random_state=0, shuffle=True)
    # models = {  
    #              "baseline": BaseLineTree(min_samples_split =4,random_state=0, max_depth=2),
    #              "sklearn": StableTree5(min_samples_split =4,ntrees = 1000 ,random_state=1, max_depth=2)
    #         }
    # stability = {name:[] for name in models.keys()}
    # standard_stability = {name:[] for name in models.keys()}
    # times = {name:[] for name in models.keys()}
    # mse = {name:[] for name in models.keys()}
    # iteration = 1


    # for train_index, test_index in kf.split(X):
    #     X_12, y_12 = X[train_index],y[train_index]
    #     X_test,y_test = X[test_index],y[test_index]
    #     X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=0)
    #     for name, model in models.items():
    #         start = time.time()
    #         model.fit(X1,y1)
            

    #         pred1 = model.predict(X_test)
    #         model.update(X_12, y_12)
    #         pred2 = model.predict(X_test)
    #         end = time.time() 
    #         mse[name].append(mean_squared_error(y_test,pred2))
    #         times[name].append(end-start)
    #         stability[name].append(np.log((pred1.item()+1e-3)/(pred2.item()+1e-3)))
    #         standard_stability[name].append(abs(pred1.item()- pred2.item()))
        
    #         if (iteration) % 100 ==0:
    #             print(f"{iteration}/{N}, {name} - mse: {np.mean(mse[name]):.3f}, stability: {np.std(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}, time: {np.mean(times[name]):.3f}")

    #     iteration+=1

    # print(models)
    # for name in models.keys():
    #     print("="*80)
    #     print(f"{name} - mse: {np.mean(mse[name]):.3f}, stability: {np.std(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}, time: {np.mean(times[name]):.3f}")
    #     print("="*80)   


    #tree.plot()
    


  
  
    # X  = np.array([[1,1],[1,2],[2,4],[2,3],[4,5], [2.5,3.5], [6,10], [4,6] ])
    # y = X[:,0]*2 + X[:,1]*0.5 
    # start = time.time()
    # tree.update(X,y)
    # mse = mean_squared_error(y,tree.predict(X))
    # end = time.time()
    # print(f"my impl: time {end - start}, mse {mse}" )
    # tree.plot()

 

    

    
    
   
            