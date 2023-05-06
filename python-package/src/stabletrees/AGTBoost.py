from _stabletrees import agtboost 
import numpy as np
class AGTBoost():

    def __init__(self,loss_function : str = "mse", nrounds:int = 5000, learning_rate:float = 0.01, gamma = 0.5) -> None:
        self.loss_function = loss_function
        self.nrounds = nrounds
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = agtboost()
        self.model.set_param(self.nrounds,self.learning_rate,0,self.loss_function,self.gamma)
    

    def fit(self,X : np.ndarray ,y : np.ndarray,verbose: int = 0, sample_weight: np.ndarray = None, offset: np.ndarray = None):
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        if offset is None:
            offset = np.zeros(shape=(len(y),))
        self.model.learn(y,X,verbose,False,False, sample_weight, offset)
        return 
    

    def update(self, X : np.ndarray ,y : np.ndarray,verbose: int = 0, sample_weight: np.ndarray = None, offset: np.ndarray = None):
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        if offset is None:
            offset = np.zeros(shape=(len(y),))
        prev_pred = self.predict(X)
        self.model.update(y,prev_pred,X,verbose,False,False, sample_weight,offset)
        return 
    
    def predict(self,X : np.ndarray,offset: np.ndarray = None):
        if offset is None:
            offset = np.zeros(shape=(X.shape[0],))
        return self.model.predict(X,offset)
    

# num_tasks = 10
# num_cores = 4
# import time
# n = 1000
# X = np.random.uniform(0,4,size=(n,1))
# #y = np.random.normal(X.ravel(),1,(n,))
# y = np.random.poisson(X.ravel()**2,(n,))

# m = AGTBoost(loss_function="poisson")
# m.fit(X,y,2)
# print(m.predict(X))


# from matplotlib import pyplot as plt 

# plt.scatter(X,y)
# plt.scatter(X,np.exp(m.predict(X)))
# plt.show()
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# ds = "Boston"
# data = pd.read_csv("data/"+ ds+".csv") # load dataset

# # data preperation
# # data = data.dropna(axis=0, how="any") # remove missing values if any
# # data = data.loc[:, feature + [target]] # only selected feature and target variable
# # cat_data = data.select_dtypes("object") # find categorical features
# # if not cat_data.empty: # if any categorical features, one-hot encode them
# #     cat_data = pd.get_dummies(data.select_dtypes("object"), prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
# #     data = pd.concat([data.select_dtypes(['int','float']),cat_data],axis=1)

# #print(data.corr())
# target = "medv"
# m = AGTBoost(loss_function="mse", learning_rate=0.01, gamma=0.5)

# y = data[target].to_numpy()
# X = data.drop(target, axis=1).to_numpy()
# X_train,X_test,y_train,y_test =   train_test_split(X,y, test_size=0.25,random_state=0)

# X1,X2,y1,y2 =  train_test_split(X_train,y_train, test_size=0.25,random_state=0)

# m.fit(X1,y1,0)
# print(mean_squared_error(y_test, m.predict(X_test)))
# m2 = AGTBoost(loss_function="mse", learning_rate=0.01)
# m.fit(X_train,y_train,0)
# print(mean_squared_error(y_test, m.predict(X_test)))
# m.update(X_train,y_train,0)
# print(mean_squared_error(y_test, m.predict(X_test)))


