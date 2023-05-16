
# from joblib import Parallel, delayed
from stabletrees.tree import BaseLineTree,Node,Tree, AbuTree, TreeReevaluation, StabilityRegularization,BABUTree
# from _stabletrees import ParallelSum 
import numpy as np
# # from stabletrees.random_forest import RF

# from _stabletrees import agtboost 

# # num_tasks = 10
# # num_cores = 4
# # import time
np.random.seed(0)
X = np.random.uniform(0,4,size=(200,5))
y = np.random.normal(X[:,0] + X[:,1] * X[:,2]+ X[:,0]*0.5 ,1,(200,))
import matplotlib.pyplot as plt
# w = np.ones_like(y)
# print(w)
# p = agtboost()
# p.learn(y,X,10,True,False, w, np.zeros_like(y))
# print(p.predict(X,w),y )


t = TreeReevaluation(max_features=2, adaptive_complexity=True)

t.fit(X,y)
t.update(X,y)
t.plot()
np.random.seed(0)
X = np.random.uniform(0,4,size=(200,5))
y = np.random.normal(X[:,0] + X[:,1] * X[:,2]+ X[:,0]*0.5 ,1,(200,))

plt.show()
