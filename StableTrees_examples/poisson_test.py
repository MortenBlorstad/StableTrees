
import numpy as np  
from stabletrees import BaseLineTree,AbuTree,AbuTreeI, BABUTree,StabilityRegularization
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
n = 1000
np.random.seed(0)
X = np.random.uniform(low=0,high=4,size=(n,1))

y = np.random.poisson(lam=np.exp(X[:,0]),size=n)
print(X.shape, y.shape)
print(X, y.shape)
#y = np.random.normal(loc = X.ravel()**2,scale=1,size=  n)
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)

criterion = "poisson"



t2 =AbuTreeI(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X1,y1)
t1 = BaseLineTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X1,y1)
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
pred4 =t1.predict(X[ind,:])
pred2 =t2.predict(X[ind,:])
#print(pred)
t = BaseLineTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X,y)
pred5 =t.predict(X[ind,:])
plt.scatter(X,y, s= 10)

plt.plot(X[ind,:],pred,c ="black",label= "base") #first
#plt.scatter(X,pred3,c ="orange", s = 8) #t2
plt.plot(X[ind,:],pred2,c ="r",label= "ABU") #combined

plt.plot(X[ind,:],pred5,c ="y",label= "base retrained")

plt.legend()
plt.show()
print(pred-pred5)


# X1 = np.array([1,2,3,4]).reshape(-1,1)
# y1 = np.array([1,2,4,12])
# t = DecisionTreeRegressor(criterion=criterion,max_depth=2, min_samples_leaf=1).fit(X,y)

# plot_tree(t)
# plt.show()

# t = BaseLineTree(criterion=criterion,max_depth=2, min_samples_leaf=1).fit(X,y)
# t.plot()
# plt.show()

# np.random.seed(0)
# X = np.random.multivariate_normal([0.025 ,0.075,0.05], np.array([[1, 0.1, 0], [0.1,1, 0.2], [0,0.2,1]]), size=1000)
# def formula(X, noise = 0.1):
#     return  np.clip(2*X[:,0] + 0.1*X[:,1] + 0.75*X[:,2],)
# y = np.random.poisson(lam = formula(X))

# print(mean_poisson_deviance(y1,np.full(fill_value= np.mean(y1), shape= len(y1))))
# print(2*np.mean(y1*np.log(y1/y1.mean())- (y1-y1.mean())  ))
# t = DecisionTreeRegressor(criterion=criterion,max_depth=2).fit(X[:250,:],y[:250])
# print(mean_poisson_deviance(y,t.predict(X)))
# print(mean_squared_error(y,t.predict(X)))
# plot_tree(t)
# plt.show()





# t = BaseLineTree(criterion=criterion, adaptive_complexity=True).fit(X1,y1)
# t_mse = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X1,y1)
# #print(t.predict(X))
# #print(mean_poisson_deviance(y,np.exp(t.predict(X))))
# #print(t.predict(X))
# print( mean_poisson_deviance(y+0.001,t.predict(X)+0.001))
# print( mean_squared_error(y+0.001,t_mse.predict(X)+0.001))
# plt.subplot(1,3,1)
# plt.scatter(X,y)
# plt.scatter(X,t.predict(X))
# t.update(X,y)
# t_mse.update(X,y)
# #print("updated: ", mean_poisson_deviance(y,np.exp(t.predict(X))))
# print("updated: ", mean_poisson_deviance(y+0.001,t.predict(X)+0.001))
# print("updated: ", mean_squared_error(y+0.001,t_mse.predict(X)+0.001))
# plt.subplot(1,3,2)
# plt.scatter(X,y)
# plt.scatter(X,t.predict(X))

# t1 = AbuTreeI(criterion=criterion, adaptive_complexity=True).fit(X1,y1)

# print( mean_poisson_deviance(y+0.001,t1.predict(X)+0.001))
# t1.update(X,y)
# #print("updated: ", mean_poisson_deviance(y,np.exp(t1.predict(X))))
# print("updated: ", mean_poisson_deviance(y+0.001,t1.predict(X)+0.001))
# plt.subplot(1,3,3)
# plt.scatter(X,y)
# plt.scatter(X,t1.predict(X))
# plt.show()