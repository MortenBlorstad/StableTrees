import numpy as np
import unittest
from stabletrees import BaseLineTree


class TestMSE(unittest.TestCase):
    def get_testset_one(self):
        X = np.array([4,5,6]).reshape(-1,1)
        y = np.array([0,1,4])
        return X,y
    
   
    
    def test_impurity_and_score(self):
        X,y = self.get_testset_one()
        n = len(y)
        tree = BaseLineTree(max_depth = 1)
        tree.fit(X,y)
        root = tree.root
        correct_scores = 1/6

        self.assertTrue(root.get_impurity() == np.mean((y - y.mean())**2))
        self.assertTrue(root.get_split_score() == correct_scores)

    def test_prediction(self):
        X,y = self.get_testset_one()
        n = len(y)
        tree = BaseLineTree(max_depth = 1)
        tree.fit(X,y)
        ypred = tree.predict(X)
        y_pred_correct = np.array([0.5,0.5,4])
        self.assertTrue(np.all(ypred == y_pred_correct))
 



if __name__ == '__main__':
    unittest.main()