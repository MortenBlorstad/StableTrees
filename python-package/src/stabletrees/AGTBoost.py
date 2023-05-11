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
    
    def inverse_function(self, pred:np.ndarray ):
        if  self.loss_function  == "mse":
            return pred
        if self.loss_function  =="poisson":
            return np.exp(pred)
        
        raise Exception("error")

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
        prev_pred = self.predict(X,offset)
        self.model.update(y,prev_pred,X,verbose,False,False, sample_weight,offset)
        return 
    
    def predict(self,X : np.ndarray,offset: np.ndarray = None):
        if offset is None:
            offset = np.zeros(shape=(X.shape[0],))
        return self.inverse_function(self.model.predict(X,offset))
    

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


# from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_poisson_deviance
# from sklearn.compose import ColumnTransformer
# import tarfile
# import pandas as pd
# SEED = 0
# EPSILON = 1.1

# ## data prepocessing 
# with tarfile.open("data\poisson\\freMTPLfreq.tar.gz", "r:*") as tar:
#     csv_path = tar.getnames()[0]
#     df = pd.read_csv(tar.extractfile(csv_path), header=0)


# df["Frequency"] = df["ClaimNb"] / df["Exposure"]

# brand_to_letter = {'Japanese (except Nissan) or Korean': "F",
#                    'Fiat':"D",
#                     'Opel, General Motors or Ford':"C",
#                       'Mercedes, Chrysler or BMW': "E",
#                       'Renault, Nissan or Citroen': "A",
#                      'Volkswagen, Audi, Skoda or Seat':"B",
#                       'other':"G" }
# df.Brand = df.Brand.apply(lambda x: brand_to_letter[x])

# # glm binning based on book
# df["Density_binned"] = pd.cut(df.Density, include_lowest=True, bins=[0,40,200,500,4500,np.inf])
# df["DriverAge_binned"]  = pd.cut(df.DriverAge , bins=[17,22,26,42,74,np.inf])
# df["CarAge_binned"]  = pd.cut(df.CarAge, include_lowest=True , bins=[0,15,np.inf])
# df["brandF"] = np.where(df.Brand=="Japanese (except Nissan) or Korean","F","other")
# df["Power_glm"] = ["DEF" if p in ["d","e","f"] else "other" if p in ["d","e","f"] else "GH" for p in df.Power ]
# df.insert(len(df.columns)-1, 'Frequency', df.pop('Frequency'))


# # tree_preprocessor = ColumnTransformer(
# #     [
# #         ("categorical",
# #             OrdinalEncoder(),
# #             ["Brand", "Power", "Gas", "Region"],
# #         ),
# #         ("numeric", "passthrough", ["CarAge","DriverAge","Density"]),
# #     ],
# #     remainder="drop",
# # )

# tree_preprocessor = ColumnTransformer(
#     [
#         ("categorical",
#             OrdinalEncoder(),
#             ["Gas"],
#         ),
#         ("numeric", "passthrough", ["CarAge","DriverAge","Density"]),
#     ],
#     remainder="drop",
# )

# EPSILON =1.1
# def S1(pred1, pred2):
#     return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

# print(tree_preprocessor.fit_transform(df).shape)

# df12, df_test =  train_test_split(df, test_size=0.25, random_state=0)
# df1, df2 =  train_test_split(df12, test_size=0.5, random_state=0)

# m = AGTBoost(loss_function="mse", learning_rate=0.01, gamma=0.5)


# m.fit(tree_preprocessor.transform(df1),df1.ClaimNb,25,offset =np.log(df1.Exposure))

# pred1 = m.predict(tree_preprocessor.transform(df_test),offset = np.log(df_test.Exposure) )
# print(mean_poisson_deviance(df_test.ClaimNb, pred1))

# m.fit(tree_preprocessor.transform(df12),df12.ClaimNb,25,offset =np.log(df12.Exposure))
# pred2 = m.predict(tree_preprocessor.transform(df_test),offset = np.log(df_test.Exposure) )
# print(mean_poisson_deviance(df_test.ClaimNb, pred2))
# print(S1(pred1,pred2))