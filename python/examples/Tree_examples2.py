import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import sys
import os
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
cur_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_file_path + '\\..\\..\\cpp\\build\\Release\\')
sys.path.append(cur_file_path + '\\..')

from tree.RegressionTree import BaseLineTree, StableTree1,sklearnBase,StableTree2,StableTree3, StableTree4,StableTreeM2
parameters = {'max_depth':[None, 5, 10], 'min_samples_split':[2,4,8]}
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)

datasets = ["Boston", "Carseats","College", "Hitters", "Wage"]
targets = ["medv", "Sales", "Apps", "Salary", "wage"]
features = [[ "crim", "rm" ], ["Advertising", "Price"], ["Private", "Accept"], ["AtBat", "Hits"], ["year", "age"] ]
SEED = 0
EPSILON = 1.1
for ds,target, feature in zip(datasets,targets, features):
    iteration = 1
    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
    data = pd.read_csv("data/"+ ds+".csv")
    data = data.dropna(axis=0, how="any")
    data = data.loc[:, feature + [target]]
    cat_data = data.select_dtypes("object")
    if not cat_data.empty:
        cat_data = pd.get_dummies(data.select_dtypes("object"), prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
        data = pd.concat([data.select_dtypes(['int','float']),cat_data],axis=1)

    print(data.describe())
    y = data[target].to_numpy()
    X = data.drop(target, axis=1).to_numpy()
    X12, X_test, y12, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
    X1, X2, y1, y2 = train_test_split(X12,y12, test_size=0.5, random_state=0)
    
    # initial model 
    models = {  
                "baseline": BaseLineTree(),
                "sklearn": sklearnBase(),
                "method1":StableTree1(),
                "method2":StableTreeM2(),
        }
    stability = {name:[] for name in models.keys()}
    standard_stability = {name:[] for name in models.keys()}
    mse = {name:[] for name in models.keys()}
    train_stability = {name:[] for name in models.keys()}
    train_standard_stability = {name:[] for name in models.keys()}
    train_mse = {name:[] for name in models.keys()}
    i = 0
    for train_index, test_index in kf.split(X):
        X_12, y_12 = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
        i+=1
        clf.fit(X1,y1)
        params = clf.best_params_
        print(params)
        models = {  
                "baseline": BaseLineTree(**params),
                "sklearn": sklearnBase(**params,random_state=0),
                "method1":StableTree1(**params,delta=0.0001),
                "method2":StableTreeM2(**params),
        }
        for name, model in models.items():
            model.fit(X1,y1)

            pred1 = model.predict(X_test)
            pred1_train = model.predict(X_12)
           
            
            model.update(X_12,y_12)
            pred2 = model.predict(X_test)
            
            pred2_train =  model.predict(X_12)
            train_mse[name].append(mean_squared_error(pred2_train,y_12))
            train_stability[name].append(np.std(np.log((pred1_train+EPSILON)/(pred2_train+EPSILON))))
            train_standard_stability[name].append(np.mean(abs(pred1_train- pred2_train)))
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(np.std(np.log((pred1+EPSILON)/(pred2+EPSILON))))
            standard_stability[name].append(np.mean(abs(pred1- pred2)))
        

    print(ds)
    train_mse_denom = np.mean(train_mse["baseline"])
    train_S1_denom = np.mean(train_stability["baseline"])
    train_S2_denom = np.mean(train_standard_stability["baseline"])
    mse_denom = np.mean(mse["baseline"])
    S1_denom = np.mean(stability["baseline"])
    S2_denom = np.mean(standard_stability["baseline"])

    for name in models.keys():
        print("="*80)
        print(f"{name}")
        print(f"train - mse: {np.mean(train_mse[name])/train_mse_denom:.3f}, stability: {np.mean(train_stability[name])/train_S1_denom:.3f}, standard stability: {np.mean(train_standard_stability[name])/train_S2_denom:.3f}")
        print(f"test - mse: {np.mean(mse[name]):.3f}({np.mean(mse[name])/mse_denom:.3f}), stability: {np.mean(stability[name]):.3f}({np.mean(stability[name])/S1_denom:.3f}), standard stability: : {np.mean(standard_stability[name]):.3f}({np.mean(standard_stability[name])/S2_denom:.3f})")
        print("="*80)
    print()
    

        

