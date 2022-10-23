import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')

from stable_trees import Node, Splitter, Treebuilder


import numpy as np
import unittest

class TestKF(unittest.TestCase):
    def test_node_constructor(self):
        split_value =1.0
        prediction = 1.0
        n_samples = 1
        split_feature =1
        split_score = 1
        node= Node(split_value,split_score,split_feature,n_samples, prediction)

        self.assertTrue(node.is_leaf())
        left= Node(split_value,split_score,split_feature,n_samples, prediction)
        right= Node(split_value,split_score,split_feature,n_samples, prediction)
        node.set_left_node(left)
        node.set_right_node(right)
        self.assertFalse(node.is_leaf())
        self.assertEqual(node.predict(),prediction )

    def test_splitter_get_predictions(self):
        splitter = Splitter()
        y = np.array([1.0,1.0,0.0,0.0,2.0,2.0])
        X = np.array([0.0,0.0,1.0,1.0,2.0,3.0])
        value = 0
        mask = X<=value
        y_left = y[mask]
        y_right = y[np.invert(mask)]
       
        vals = splitter.get_predictions(X,y,value)
        self.assertEqual(vals[0:2], (np.mean(y_left),np.mean(y_right)) )
        self.assertTrue(np.all(vals[2] ==y_left ))
        self.assertTrue(np.all(vals[3] ==y_right ))
        
    def test_splitter_sse(self):
        splitter = Splitter()
        y = np.array([1,1,0,0,2,2])
        value = 0
        X = np.array([0,0,1,1,2,3])
        mask = X<=value
        y_true = y[mask]
        y_pred = np.mean(y_true)
        
        
        self.assertEqual(splitter.sum_squared_error(y_true,y_pred), np.sum((y_true-y_pred)**2) )
        
        y_true = y[np.invert(mask)]
        y_pred = np.mean(y_true)
        
        self.assertEqual(splitter.sum_squared_error(y_true,y_pred), np.sum((y_true-y_pred)**2) )

    def test_splitter_mse(self):
        splitter = Splitter()
        y = np.array([1,1,0,0,2,2])
        value = 0
        X = np.array([0,0,1,1,2,3])
        mask = X<=value

        y_true_left = y[mask]
        y_pred_left = np.mean(y_true_left)
        y_true_right = y[np.invert(mask)]
        y_pred_right = np.mean(y_true_right)
        solution = (np.sum((y_true_right-y_pred_right)**2) + np.sum((y_true_left-y_pred_left)**2))/2
        
        self.assertEqual(splitter.mse_criterion(X, y,value),  solution)

    def test_splitter_split_feature(self):
        y = np.array([0.5,0.3,0.7,1.0,2.0,2.5])
        X = np.array([0.0,0.0,0.5,1.0,2.0,3.0])
        value = X.mean()
        mask = X<=value
        y_left = y[mask]
        y_right = y[np.invert(mask)]
        splitter = Splitter()
        score,split_value = splitter.select_split(X,y)
        

        y_pred_left = np.mean(y_left)
        y_pred_right = np.mean(y_right)
        solution = (np.sum((y_right-y_pred_right)**2) + np.sum((y_left-y_pred_left)**2))/2

        self.assertEqual(score,  solution)
        self.assertEqual(split_value, value )

    def test_splitter_split_data(self):
        y = np.array([0.5,1.0,2.5])
        X = np.array([[0.0,0.0],[0,1.0],[0,3.0]])

        value = X[:,1].mean()
        mask = X[:,1]<=value
        y_left = y[mask]
        y_right = y[np.invert(mask)]
        y_pred_left = np.mean(y_left)
        y_pred_right = np.mean(y_right)
        
        solution = (np.sum((y_right-y_pred_right)**2) + np.sum((y_left-y_pred_left)**2))/2

        splitter = Splitter()
        split_feature,min_score,best_split_value =  splitter.find_best_split(X,y)
        
        self.assertEqual(split_feature,  1)
        self.assertEqual(min_score, solution )
        self.assertEqual(best_split_value, value )

    def test_TreeBuilder_stop_conditions(self):
        nrow = 100
        ncol = 10
        treebuilder = Treebuilder()
        
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


        

        
        
        
if __name__ == '__main__':
    unittest.main()