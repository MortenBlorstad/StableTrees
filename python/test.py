import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')

from stable_trees import Node, Splitter, Tree
from tree.RegressionTree import BaseLineTree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression,load_diabetes

import numpy as np
import unittest

def is_same_on_train(X,y):
    sk = DecisionTreeRegressor(min_samples_split =2, random_state=0).fit(X,y)
    skmse = mean_squared_error(y,sk.predict(X))
    tree = BaseLineTree(min_samples_split =2,random_state=0).fit(X,y)
    mse = mean_squared_error(y,tree.predict(X))
    return skmse==mse

def is_same_on_test(X,y):
    X1,X2,y1,y2 =train_test_split(X,y,test_size=0.2,random_state=0)

    sk = DecisionTreeRegressor(min_samples_split =3,random_state=0).fit(X1,y1)
    skmse = mean_squared_error(y2,sk.predict(X2))
    tree = BaseLineTree(min_samples_split =2,random_state=0).fit(X1,y1)
    mse = mean_squared_error(y2,tree.predict(X2))
    print("diff",skmse-mse, skmse,mse)
    return skmse==mse
    

class TestKF(unittest.TestCase):
    def test_TreeBuilder_stop_conditions(self):
        nrow = 100
        ncol = 10
        treebuilder = Tree(1000,2)
        
        y_same = np.ones(shape=nrow)
        y_diff = np.random.uniform(size=nrow, low=0, high = 1)
        self.assertTrue(treebuilder.all_same(y_same))
        self.assertFalse(treebuilder.all_same(y_diff))
        X_same = np.ones(shape=(nrow,ncol))
        X_diff = np.random.uniform(size=(nrow,ncol), low=0, high = 1)
        self.assertTrue(treebuilder.all_same_features_values(X_same))
        self.assertTrue(treebuilder.all_same_features_values(X_same))
        X_same[:,0] = np.zeros(shape=nrow)
        self.assertTrue(treebuilder.all_same_features_values(X_same))
        X_same[:,0] = np.random.uniform(size=nrow, low=0, high = 1)
        self.assertFalse(treebuilder.all_same_features_values(X_same))
    
    def test_same_as_sklearn_diabetes(self):
        X,y = load_diabetes(return_X_y=True)
        self.assertTrue(is_same_on_train(X,y))
        self.assertTrue(is_same_on_test(X,y))
    # def test_same_as_sklearn_make_regression(self):
    #     N = 500
    #     X,y= make_regression(N, 10, random_state=10)
    #     y = y + np.max(y)+100
    #     self.assertTrue(is_same_on_train(X,y))
    #     #self.assertTrue(is_same_on_test(X,y))

    def test_same_as_sklearn_simple_data(self):
        X  = np.array([[1.1,1],[1,2],[2.1,4],[2,3],[4,5],[6,10], [4,6]])
        y = X[:,0]*2 + X[:,1]*0.5 
        self.assertTrue(is_same_on_train(X,y))
        #self.assertTrue(is_same_on_test(X,y))


    

    
        
        
    


  
        
if __name__ == '__main__':
    unittest.main()