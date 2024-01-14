# Usage
import numpy as np    
from stabletrees import ABUForest
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
np.random.seed(0)
n =10
noise = 2
X_train = np.random.uniform(0,4,(n,2))
y_train = np.random.normal(X_train.sum(axis=1),noise,n)



pca = PCA(n_components=1)
np.random.seed(0)
n =1000
noise = 2
X = np.random.uniform(0,4,(n,2))
y = np.random.normal(X.sum(axis=1),noise,n)

X_test = np.sort(np.random.uniform(0,4,(n,2)),axis=0)
y_test = np.random.normal(X_test.sum(axis=1),noise,n)

X1,_,y1,_ =  train_test_split(X,y,train_size=0.5)

pca.fit(X1)
pca1 = pca.transform(X_test)

abu = ABUForest(n_estimators=50, random_state=1)

rf = RandomForestRegressor(n_estimators=50, random_state=1, max_leaf_nodes=5)

abu.fit(X1,y1)
rf.fit(X1,y1)



fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

pred1 = abu.predict(X_test)
pred_rf1 = rf.predict(X_test)

abu.update(X,y)
pred2 = abu.predict(X_test)

rf.fit(X,y)
pred_rf2 = rf.predict(X_test)

ax1.scatter(pca1,y_test,c = "red", s = 2, alpha = 0.5)
ax1.set_ylabel("y")
ax1.set_xlabel("pca1")
ax1.set_title("trained at t=0")
ax1.plot(pca1,pred1,c = "black")
ax1.plot(pca1,pred_rf1,c = "green")
ax2.scatter(pca1,y_test,c = "red", s = 2, alpha = 0.5)
ax2.set_ylabel("y")
ax2.set_xlabel("pca1")
ax2.set_title("updated at t=1")
ax2.plot(pca1,pred2,c = "black")
ax2.plot(pca1,pred_rf2,c = "green")
plt.show()