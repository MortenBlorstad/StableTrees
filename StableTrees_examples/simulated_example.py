from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from stabletrees.tree import Tree,ABUTree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

pca = PCA(n_components=1)

n =1000
X = np.random.uniform(0,4,(n,1))
y = np.random.normal(X.ravel()**2,1,n)

X_test = np.sort(np.random.uniform(0,4,(n,1)),axis=0)
y_test = np.random.normal(X_test.ravel()**2,1,n)

X1,_,y1,_ =  train_test_split(X,y,train_size=0.5)



abu = ABUTree(adaptive_complexity=True,min_samples_leaf=5)

abu.fit(X1,y1)

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

pred1 = abu.predict(X_test)
abu.update(X,y)
pred2 = abu.predict(X_test)

ax1.scatter(X_test,y_test,c = "red")
ax1.plot(X_test,pred1,c = "black")
ax2.scatter(X_test,y_test,c = "red")
ax2.plot(X_test,pred2,c = "black")
plt.show()

