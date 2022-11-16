import numpy as np 
class Node:
    def __init__(self,score = None,feature_index = None,split_value=None, prediction = None) -> None:
        self.prediction = prediction
        self.score = score
        self.feature_index = feature_index
        self.split_value = split_value
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        return self.left_child is None and self.right_child is None
        


    

class tree:
    def __init__(self) -> None:
        self.root = None
        self.lmbda = 0.01
        self.max_depth = np.iinfo(np.intp).max
        self.min_samples_split = 1

    def split(self, X : np.ndarray,y:np.ndarray, y_prev_prediction:np.ndarray,lmbda :float):
        min_score = -np.inf
        feature_index = 0
        X_indices_sorted = np.argsort(X, axis=0)
        rows,cols = X.shape
        split_value = 0
        best_mask = None
        G = self.firstDerivative(y_prev_prediction,np.mean(y),y)
        H = self.secondDerivative(len(y))
        for c in range(cols):
            feature = X[:, c]
    
            
            for r in range(0,rows-1):
                l = X_indices_sorted[r,c]
                u = X_indices_sorted[min(r+1,rows-1),c]
          
                
                value = (feature[l]+feature[u])/2
                
                mask = feature<=value
                
                y_mean_left = np.mean(y[mask])
                y_mean_right = np.mean(y[np.invert(mask)])
                G_L= self.firstDerivative(y_prev_prediction[mask],y_mean_left,y[mask])
                H_L= self.secondDerivative(mask.sum())
                G_R = G- G_L
                H_R = H- H_L
                
                
                score = G_L**2/(H_L+lmbda)+G_R**2/(H_R+lmbda)- G**2/(H+lmbda) 
                
                
                if min_score<score:
                    min_score =  score
                    
                    feature_index = c
                    split_value = value
                    best_mask = mask
                    #print(score,min_score, best_mask, "sad")

        return best_mask,feature_index,min_score, split_value 

    def firstDerivative(self,y_bar1: float,y_bar2:float, y: np.ndarray)->float:
        return (-2*np.sum(y-y_bar2) - 0.1*2*np.sum(y_bar1-y_bar2))


    def secondDerivative(self,n)->float:
        return (2+0.1*2)*n

    def __all_same_label(self,y):
        return np.all(y[0]==y)
    def __all_same_features_values(self,X):
        return np.all(np.apply_along_axis(lambda x:len(np.unique(x)) , 0, X) ==1)

    def build(self,X,y,depth = 0, prev_prediction=None):
        y_mean = np.mean(y)
        if depth >=self.max_depth:
            return Node(prediction=y_mean)

        if  len(y)<=self.min_samples_split:
            return Node(prediction=y_mean)
            
        if  len(y)<=1:
            return Node(prediction=y_mean)

        if self.__all_same_label(y):
            return Node(prediction=y_mean)

        if self.__all_same_features_values(X):
            return Node(prediction=y_mean)


        if prev_prediction is None:
            prev_pred = np.full(shape= y.shape,fill_value= y_mean)
        else:
            prev_pred = prev_prediction

        
        
        mask_left,feature_index, score, split_value = self.split(X,y,prev_pred,self.lmbda)
       
        
        mask_right = np.invert(mask_left)
        if  mask_left.sum()<=self.min_samples_split:
            return Node(prediction=y_mean)
        if mask_right.sum()<=self.min_samples_split:
            return Node(prediction=y_mean)

        node = Node(score,feature_index,split_value, y_mean)

        if prev_prediction is None:
            prev_pred_left = np.full(shape = mask_left.sum(), fill_value= np.mean(y[mask_left]))
            prev_pred_right = np.full(shape =mask_right.sum(), fill_value=np.mean(y[mask_right]))
        else:
            prev_pred_left = prev_prediction[mask_left]
            prev_pred_right = prev_prediction[mask_right]

        

        node.right_child = self.build(X[mask_right],y[mask_right],depth+1,prev_pred_right)

        node.left_child = self.build(X[mask_left],y[mask_left],depth+1,prev_pred_left)

        return node
    def fit(self,X,y):
        self.root = self.build(X,y)
        return self


    def __predict(self,X):
        node = self.root
        while not node.is_leaf():
            if X[node.feature_index]<=node.split_value:
                node = node.left_child
            else:
                node = node.right_child
        return node.prediction


    def predict(self, X):
        predictions =[]
        for row in range(X.shape[0]): # iterate over all data points
            x = X[row,:]
            prediction = self.__predict(x) # predict data point
            predictions.append(prediction) # add predicton to the list of predictions.
        return np.array(predictions)

    def update(self,X,y):
        y_pred = self.predict(X)
        self.root = self.build(X,y,depth=0,prev_prediction= y_pred)
        return self


                


            




        



from sklearn.datasets import make_regression,load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,KFold
from matplotlib import pyplot as plt



import pandas as pd
N = 2000
dataset_name =  "make_regression"  #"sim_claim_freq" #"diabetes" #

X,y= make_regression(N,10, random_state=0)

y = y + np.max(y)+100
# y +=max(np.abs(y))
# y = np.log(y+1)
# X,y = load_diabetes(return_X_y=True)
# N = len(y)
X1,X2,y1,y2 =  train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.tree import DecisionTreeRegressor
import sys
import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append("C:\\Users\\mb-92\\OneDrive\\Skrivebord\\studie\\StableTrees\\cpp\\build\\Release")
sys.path.append(cur_file_path + '\\python')
from tree.RegressionTree import BaseLineTree, StableTree1
t = tree().fit(X1,y1)
print(mean_squared_error(y1,t.predict(X1)))
skt = DecisionTreeRegressor(random_state=0, min_samples_split=5).fit(X1,y1)
print(mean_squared_error(y1,skt.predict(X1)))
skt = BaseLineTree(min_samples_split=5).fit(X1,y1)
print(mean_squared_error(y1,skt.predict(X1)))

# kf = KFold(n_splits=X.shape[0], random_state=0, shuffle=True)
    
# models = {  
#             # "baseline": BaseLineTree(min_samples_split=min_samples_split),
#             # "sklearn": sklearnBase(min_samples_split=min_samples_split, random_state=0),
#             # "method1":StableTree1(delta=0.0001, min_samples_split=min_samples_split),
#             # "method2":StableTree2(min_samples_split=min_samples_split),
#             # "method3":StableTree3(min_samples_split=min_samples_split),
#             "method4":tree()
#         }

# stability = {name:[] for name in models.keys()}
# standard_stability = {name:[] for name in models.keys()}
# mse = {name:[] for name in models.keys()}
# iteration = 1
# for train_index, test_index in kf.split(X):
#     X_12, y_12 = X[train_index],y[train_index]
#     X_test,y_test = X[test_index],y[test_index]
#     X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=0)

    
#     # initial model 
    
#     for name, model in models.items():
            
#         model.fit(X1,y1)
    
#         pred1 = model.predict(X_test)
#         model.update(X_12,y_12)
#         pred2 = model.predict(X_test)
        
#         if(np.isnan(pred2).any()):
#             print(y_test,pred2, X_test.shape, X_test)

#         mse[name].append(mean_squared_error(y_test,pred2))
        
#         stability[name].append(np.log((pred1.item()+1e-3)/(pred2.item()+1e-3)))
#         standard_stability[name].append(abs(pred1.item()- pred2.item()))
    
#         if (iteration) % 50 ==0:
#             print(f"{iteration}/{N}, {name} - mse: {np.mean(mse[name]):.3f}, stability: {np.std(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")

#     iteration+=1

# print(models)
# for name in models.keys():
#     print("="*80)
#     print(f"{name} - mse: {np.mean(mse[name]):.3f}, stability: {np.std(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")
#     print("="*80) 


# mse1 =[]
# y_hat1 = []
# for train_index, test_index in kf1.split(X1):
#     X_train,y_train = X1[train_index],y1[train_index]
#     X_holdout,y_holdout = X1[test_index],y1[test_index]
#     y_bar = np.mean(y_train)
#     y_hat1.append(y_bar)
#     mse1.append((y_holdout.item()-y_bar)**2)

# y_hat2 = [] 

# def dloss(y_bar1: float,y_bar2:float, y: np.ndarray,w:float, lmbda:float)->float:
#     return (-2*np.mean(y-y_bar2*(1+w))- (y_bar1-y_bar2*(1+w)))*y_bar2 + 2*lmbda*w





# mse2 = []
# for train_index, test_index in kf2.split(X2):
#     X_train,y_train = X2[train_index],y2[train_index]
#     X_train = np.vstack((X_train,X1))
#     y_train = np.concatenate((y_train,y1), axis=0)
#     X_holdout,y_holdout = X2[test_index],y2[test_index]
#     y_bar = np.mean(y_train)
#     y_hat2.append(y_bar)
#     mse2.append((y_holdout.item()-y_bar)**2)
    
    
# y_hat3 = [] 
# mse3 = []
# kf3 = KFold(n_splits=X12.shape[0], random_state=0, shuffle=True)   
# for train_index, test_index in kf3.split(X12):
#     w = 0
#     y_bar1 = np.mean(y_hat1)
#     y_bar2 = np.mean(y_hat2)
#     X_holdout,y_holdout = X12[test_index],y12[test_index]
#     for epoch in range(20):
#         w -= 0.001*dloss(y_bar1,y_bar2,y12[train_index],w, lmbda = 0.1)
#     print((1+w))
#     mse3.append((y_holdout.item()-y_bar*(1+w))**2)
#     y_hat3.append(y_bar2*(1+w))

# plt.boxplot([y_hat1,y_hat2,y_hat3])
# labels = ["t0", "t1", "t1_adjusted"]
# plt.xticks(range(1, len(labels) + 1), labels)
# plt.ylabel(r"$\hat{y}$")
# plt.show()

# print(np.mean(mse1))
# print(np.mean(mse2))
# print(np.mean(mse3))
