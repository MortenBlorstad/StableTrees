
from joblib import Parallel, delayed
from stabletrees.tree import BaseLineTree,Node,Tree
from _stabletrees import ParallelSum 
import numpy as np
from stabletrees.random_forest import RF

num_tasks = 10
num_cores = 4
import time

X = np.random.uniform(0,4,size=(1000,1))
y = np.random.normal(X.ravel(),1,(1000,))
w = np.ones_like(y)
p = RF()
p.fit(X,y)
p.predict(X)
# start_time = time.time()
# for i in range(1):
#     X = np.random.uniform(0,4,size=(1000,1))
#     y = np.random.normal(X.ravel(),1,(1000,))
#     w = np.ones_like(y)
#     p = ParallelSum(y)

#     print(p.learn(X,y,w ).shape)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time: ", execution_time, " seconds")
# start_time = time.time()
# X = np.random.uniform(0,4,size=(10000,2))
# y = np.random.normal(X[:,0] +X[:,0] /X[:,1] ,1,(10000,))

# t = Tree(0, 50,  5.0, 5,  True,  1 ,1,  0)
# w = np.ones_like(y)
# print(w)
# t.learn(X,y,w)
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time: ", execution_time, " seconds")
# start_time = time.time()
# for i in range(1):
#     X = np.random.uniform(0,4,size=(10000,2))
#     y = np.random.normal(X[:,0] +X[:,0] /X[:,1] ,1,(10000,))
#     w = np.ones_like(y)
#     p = ParallelSum(y)

#     p.learnslow(X,y, w )

# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time: ", execution_time, " seconds")
# Node(0.1, 1, 1, 1, 1, 1, 1, 1, [1] )

# print("sad")

# Parallel(n_jobs=num_cores)(delayed(Node(0.1, 1, 1, 1, 1, 1, 1, 1, [1] ))(1000000) for _ in range(num_tasks))