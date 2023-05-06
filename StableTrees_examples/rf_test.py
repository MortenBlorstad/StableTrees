
# from joblib import Parallel, delayed
# from stabletrees.tree import BaseLineTree,Node,Tree
# from _stabletrees import ParallelSum 
import numpy as np
# from stabletrees.random_forest import RF

from _stabletrees import agtboost 

# num_tasks = 10
# num_cores = 4
# import time

X = np.random.uniform(0,4,size=(10,1))
y = np.random.normal(X.ravel(),1,(10,))

w = np.ones_like(y)
print(w)
p = agtboost()
p.learn(y,X,10,True,False, w, np.zeros_like(y))
print(p.predict(X,w),y )