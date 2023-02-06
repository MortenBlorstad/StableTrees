import numpy as np
import unittest
from _stabletrees import MSE, Poisson, MSEReg, PoissonReg


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


    def test_init_with_reg(self):
        X,y = self.get_testset_one()
        yprev = np.array([0.25,0.25,4])
        n = len(y)
        mse = MSEReg()
        mse.init(n,y,yprev)
        self.assertTrue(mse.get_score()==0)
        self.assertTrue(mse.node_impurity(y)== np.mean((y - y.mean())**2))

    def test_update(self):
        X,y = self.get_testset_one()
        yprev = np.array([0.25,0.25,4])
        mse = MSEReg()
        n = len(y)
        mse.init(n,y,yprev)
        correct_scores = [round(1.5+2.4583,4), round(0.16667+0.0416667,4)]
        for i, (y_i,yp_i) in enumerate(zip(y,yprev)):
            mse.update(y_i,yp_i)
            score = np.round(mse.get_score(),4)
            if i <len(correct_scores):
                self.assertTrue(score == correct_scores[i])
        


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
        self.assertTrue(np.round(crit.node_impurity(y),4)== np.round(2*np.mean(y*np.log(y/y.mean())- (y-y.mean())  ),4 ))

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

    def test_init_with_reg(self):
        X,y = self.get_testset_one()
        n = len(y)
        yprev = np.array([1.25,1.25,4])
        crit = PoissonReg()
        crit.init(n,y,yprev)
        self.assertTrue(crit.get_score()==0)
        self.assertTrue(np.round(crit.node_impurity(y),4)== np.round(2*np.mean(y*np.log(y/y.mean())- (y-y.mean())  ),4 ))

    def test_update_with_reg(self):
        X,y = self.get_testset_one()
        yprev = np.array([1.25,1.25,4])
        crit = PoissonReg()
        n = len(y)
        crit.init(n,y,yprev)
        correct_scores = [round(0.1133+0.27844,4),round(0.056633+0.01473333,4)]
        for i, (y_i,yp_i) in enumerate(zip(y, yprev)):
            if i <len(correct_scores):
                crit.update(y_i,yp_i)
                score = np.round(crit.get_score(),4)
                self.assertTrue(score <=correct_scores[i])
        

if __name__ == '__main__':
    unittest.main()