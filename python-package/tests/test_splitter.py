import numpy as np
import unittest
from _stabletrees import Splitter


class TestMSE(unittest.TestCase):
    def get_testset_one(self):
        X = np.array([4,5,6]).reshape(-1,1)
        y = np.array([0,1,4])
        return X,y
    
   
    
    def test_init(self):
        X,y = self.get_testset_one()
        n = len(y)
       
        splitter = Splitter(0,False)
        
        feature_index, impurity, score, split_value = splitter.find_best_split(X,y)
        
        self.assertTrue(feature_index==0)
        self.assertTrue(impurity== np.mean((y - y.mean())**2))
        self.assertTrue(score==1/6)


if __name__ == '__main__':
    unittest.main()
