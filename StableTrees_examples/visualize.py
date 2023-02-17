from stabletrees import BaseLineTree,TreeReevaluation
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes,make_regression,load_boston
from matplotlib import pyplot as plt

X,y = make_regression(2000,10,random_state=0, noise=10)
y = y + np.abs(np.min(y))

#X,y = load_diabetes(return_X_y=True)
# print(X, len(y))


# np.random.seed(0)
# X = np.random.uniform(size = (2000,2) )
# y = np.random.normal(0.75,0.1, size=2000)*X[:,1]*X[:,0] - np.random.normal(0.5,0.1, size=2000)*X[:,1] + X[:,1]*X[:,0] +  np.random.normal(0,2, size=2000)

X1,X2, y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)

# X1 = np.array([4,5,6]).reshape(-1,1)
# y1 = np.array([0,1,4])
tree = TreeReevaluation( delta=0.1,max_depth=3)
tree.fit(X1,y1)
print( tree.root.get_impurity(), tree.root.get_split_score())


tree.update(X,y)
tree.plot()
plt.show()


print(tree.tree.get_mse_ratio())
print(tree.tree.get_eps())
print(tree.tree.get_obs())
plt.plot(tree.tree.get_mse_ratio(), label = "mse ratio")
plt.plot([i for i in tree.tree.get_eps()], label = "1+eps")

plt.ylim((-0.1,2))
ind = [i for i in range(0,len(tree.tree.get_mse_ratio()),1)]
plt.xticks(ind,
    [tree.tree.get_obs()[i]  for i in ind])
plt.legend()
plt.show()