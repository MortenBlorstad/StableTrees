from matplotlib import pyplot as plt
import os

from sklearn.metrics import mean_squared_error

cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')
from stable_trees import Node

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
y = y + np.max(y)+100
X,y = load_diabetes(return_X_y=True)
#X = np.atleast_2d(X[:,0]).reshape(-1,1)
from stable_trees import ProbabalisticSplitter

ps = ProbabalisticSplitter(0)
count = [0]*10
mse = {}
seed =0
n = 10000
for _ in range(10000):
    i,score,_ = ProbabalisticSplitter(seed).find_best_split(X,y)
    seed+=1
    seed = seed % 10000
    if i not in mse.keys():
        mse[i]=score
    count[i]+=1

print(np.array(count)/n)
mse = sorted(mse.items())
values = np.array([1/(v**2) for _,v in mse])
total = values.sum()
print(np.round(values/total,3))
print(mse)



