
import numpy as np  
from stabletrees import BaseLineTree,AbuTree,AbuTreeI
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
df = pd.read_csv("C:\\Users\\mb-92\\OneDrive\\Skrivebord\\studie\\StableTrees\\StableTrees_examples\\test_data.csv")
X = df["x"].to_numpy().reshape(-1,1)
y = np.exp(df["y"].to_numpy())+100
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=1)
train_test_split
criterion = "poisson"

# X1 = np.array([1,2,3,4]).reshape(-1,1)
# y1 = np.array([1,2,4,12])
# t = DecisionTreeRegressor(criterion=criterion,max_depth=2, min_samples_leaf=1).fit(X,y)

# plot_tree(t)
# plt.show()

# t = BaseLineTree(criterion=criterion,max_depth=2, min_samples_leaf=1).fit(X,y)
# t.plot()
# plt.show()

np.random.seed(0)
X = np.random.multivariate_normal([0.025 ,0.075,0.05], np.array([[1, 0.1, 0], [0.1,1, 0.2], [0,0.2,1]]), size=1000)
def formula(X, noise = 0.1):
    return  np.exp(2*X[:,0] + 0.1*X[:,1] + 0.75*X[:,2] + np.random.normal(0,noise)) +100
y = formula(X) + 100

print(mean_poisson_deviance(y1,np.full(fill_value= np.mean(y1), shape= len(y1))))
print(2*np.mean(y1*np.log(y1/y1.mean())- (y1-y1.mean())  ))
t = DecisionTreeRegressor(criterion=criterion,max_depth=2).fit(X[:250,:],y[:250])
print(mean_poisson_deviance(y,t.predict(X)))
print(mean_squared_error(y,t.predict(X)))
plot_tree(t)
plt.show()

t = BaseLineTree(criterion=criterion, max_depth=2).fit(X[:250,:],y[:250])
print(mean_poisson_deviance(y,t.predict(X)))
print(mean_squared_error(y,t.predict(X)))
# t.plot()
# plt.show()


t = AbuTreeI(criterion=criterion).fit(X[:250,:],y[:250])
print(mean_poisson_deviance(y,t.predict(X)))
t.plot()
plt.show()

# print(mean_squared_error(y,t.predict(X)))
# ypred = np.round(t.predict(X),decimals=5)
# print(2*np.mean(ypred- np.log(ypred)*y + np.log(y)*y - y))
# print(np.unique(ypred))
# plt.scatter(X[:,0],y, alpha = 0.1)

# plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
# plt.show()
