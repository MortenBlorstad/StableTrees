from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from stabletrees import ABUTree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

pca = PCA(n_components=1)
np.random.seed(0)
n =100
noise = 2
X = np.random.uniform(0,4,(n,2))
y = np.random.normal(X.sum(axis=1),noise,n)

X_test = np.sort(np.random.uniform(0,4,(n,2)),axis=0)
y_test = np.random.normal(X_test.sum(axis=1),noise,n)

X1,_,y1,_ =  train_test_split(X,y,train_size=0.5)

pca.fit(X1)
pca1 = pca.transform(X_test)

abu = ABUTree(adaptive_complexity=False,min_samples_leaf=5)
bootstrap_idx = np.random.randint(0,X1.shape[0], X1.shape[0])

# print( X1[bootstrap_idx], X1)
# print( y1[bootstrap_idx], y1)
abu.fit(X1[bootstrap_idx],y1[bootstrap_idx])


fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

pred1 = abu.predict(X_test)
abu.update(X,y)
pred2 = abu.predict(X_test)

ax1.scatter(pca1,y_test,c = "red", s = 2, alpha = 0.5)
ax1.set_ylabel("y")
ax1.set_xlabel("pca1")
ax1.set_title("trained at t=0")
ax1.plot(pca1,pred1,c = "black")
ax2.scatter(pca1,y_test,c = "red", s = 2, alpha = 0.5)
ax2.set_ylabel("y")
ax2.set_xlabel("pca1")
ax2.set_title("updated at t=1")
ax2.plot(pca1,pred2,c = "black")
plt.show()

