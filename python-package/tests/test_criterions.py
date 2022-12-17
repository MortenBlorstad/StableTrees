import numpy as np
import unittest
from _stabletrees import MSE,Poisson


class TestMSE(unittest.TestCase):
    def get_testset_one(self):
        X = np.array([4,5,6]).reshape(-1,1)
        y = np.array([0,1,4])
        return X,y
    
   
    
    def test_init(self):
        X,y = self.get_testset_one()
        n = len(y)
        mse = MSE()
        mse.init(n,y)
        self.assertTrue(mse.get_score()==0)
        self.assertTrue(mse.node_impurity(y)== np.mean((y - y.mean())**2))

    def test_update(self):
        X,y = self.get_testset_one()
        mse = MSE()
        n = len(y)
        mse.init(n,y)
        correct_scores = [1.5,0.1667]
        for i, y_i in enumerate(y):
            mse.update(y_i)
            score = np.round(mse.get_score(),4)
            if i <len(correct_scores):
                self.assertTrue(score ==correct_scores[i])

    
        

class TestPoisson(unittest.TestCase):
    def get_testset_one(self):
        X = np.array([4,5,6]).reshape(-1,1)
        y = np.array([1,2,4])
        return X,y

    def test_init(self):
        X,y = self.get_testset_one()
        n = len(y)
        crit = Poisson()
        crit.init(n,y)
        self.assertTrue(crit.get_score()==0)
        self.assertTrue(crit.node_impurity(y)== 2*np.mean(y*np.log(y/y.mean())- (y-y.mean())  ))

    def test_update(self):
        X,y = self.get_testset_one()
        crit = Poisson()
        n = len(y)
        crit.init(n,y)
        correct_scores = [0.1133,0.0566]
        for i, y_i in enumerate(y):
            crit.update(y_i)
            score = np.round(crit.get_score(),4)
            if i <len(correct_scores):
                self.assertTrue(score ==correct_scores[i])
        

if __name__ == '__main__':
    unittest.main()