

import numpy as np
from stabletrees.random_forest import StackedRF
from sklearn.metrics import mean_squared_error


# num_tasks = 10
# num_cores = 4
# import time
np.random.seed(0)
X = np.random.uniform(0,4,size=(1000,1))
y = np.random.normal(X.ravel(),1,(1000,))

X_test = np.random.uniform(0,4,size=(1000,1))
y_test = np.random.normal(X.ravel(),1,(1000,))


p = StackedRF(learning_rate=0.01,gamma=1)
p.fit(X[:500],y[:500])
pred1 = p.predict(X_test)
print(mean_squared_error(p.predict(X_test),y_test ))

p.update(X,y)
pred2 = p.predict(X_test)
print(mean_squared_error(p.predict(X_test),y_test ))

print(mean_squared_error(pred1,pred2))