
import numpy as np  
from stabletrees import BaseLineTree,AbuTree,AbuTree, BABUTree,StabilityRegularization
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
n = 400
np.random.seed(0)
X = np.random.uniform(low=0,high=4,size=(n,1))

y = np.random.poisson(lam=np.exp(X[:,0]),size=n)

#y = np.random.normal(loc = X.ravel()**2,scale=1,size=  n)
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)

criterion = "mse"



t2 =AbuTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X1,y1)
t1 = BaseLineTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X1,y1)
t3 = StabilityRegularization(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5,gamma = 0.1).fit(X1,y1)
ind = np.argsort(X, axis=0).ravel()
pred = t1.predict(X[ind,:])
# t1.update(X,y)

print(y.sum())
print(y1.sum())
# print(mean_poisson_deviance(y,t.predict(X)))

# print(mean_poisson_deviance(y,t1.predict(X)))
# print(pred- pred2)

#t1.update(X2,y2)
#pred3 =t1.predict(X)
t1.update(X,y)
t2.update(X,y)
t3.update(X,y)
pred4 =t1.predict(X[ind,:])
pred2 =t2.predict(X[ind,:])
pred6 =t3.predict(X[ind,:])
#print(pred)
t = BaseLineTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X,y)
pred5 =t.predict(X[ind,:])
plt.scatter(X[:,0],y, s= 10)

plt.scatter(X[ind,0],pred,c ="black",label= "base") #first
#plt.scatter(X,pred3,c ="orange", s = 8) #t2
plt.scatter(X[ind,0],pred2,c ="r",label= "ABU") #combined
plt.scatter(X[ind,0],pred6,c ="m",label= "SL",s = 8) #combined

#plt.plot(X[ind,:],pred5,c ="y",label= "base retrained")

plt.legend()
plt.show()
# print(pred-pred5)


