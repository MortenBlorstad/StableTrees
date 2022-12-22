from stabletrees import BaseLineTree, StableTree1,StableTree2
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd


SEED = 0
EPSILON = 1.1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))

def S2(pred1, pred2):
    return np.mean(abs(pred1- pred2))

parameters = {'max_depth':[None, 5, 10], 'min_samples_split':[2,4,8]}
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)

# from examples in R package ISLR2, https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf
datasets = ["Boston", "Carseats","College", "Hitters", "Wage"]
targets = ["medv", "Sales", "Apps", "Salary", "wage"]
features = [[ "crim", "rm" ], ["Advertising", "Price"], ["Private", "Accept"], ["AtBat", "Hits"], ["year", "age"] ]
SEED = 0
EPSILON = 1.1
for ds,target, feature in zip(datasets,targets, features):
    iteration = 1
    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
    data = pd.read_csv("data/"+ ds+".csv") # load dataset
    
    # data preperation
    data = data.dropna(axis=0, how="any") # remove missing values if any
    data = data.loc[:, feature + [target]] # only selected feature and target variable
    cat_data = data.select_dtypes("object") # find categorical features
    if not cat_data.empty: # if any categorical features, one-hot encode them
        cat_data = pd.get_dummies(data.select_dtypes("object"), prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
        data = pd.concat([data.select_dtypes(['int','float']),cat_data],axis=1)

    print(data.describe())
    y = data[target].to_numpy()
    X = data.drop(target, axis=1).to_numpy()
   
    # initial model 
    models = {  
                 "baseline": BaseLineTree(),
                 "method1":StableTree1(delta=0.25),
                 "method2":StableTree2()
            }
    stability = {name:[] for name in models.keys()}
    standard_stability = {name:[] for name in models.keys()}
    mse = {name:[] for name in models.keys()}
    train_stability = {name:[] for name in models.keys()}
    train_standard_stability = {name:[] for name in models.keys()}
    train_mse = {name:[] for name in models.keys()}
    orig_stability = {name:[] for name in models.keys()}
    orig_standard_stability = {name:[] for name in models.keys()}
    orig_mse = {name:[] for name in models.keys()}
    for train_index, test_index in kf.split(X):
        X_12, y_12 = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
        clf.fit(X1,y1)
        params = clf.best_params_
        # initial model 
        models = {  
                 "baseline": BaseLineTree(**params),
                 "method1":StableTree1(**params,delta=0.25),
                 "method2":StableTree2(**params)
                }
        for name, model in models.items():
            model.fit(X1,y1)

            pred1 = model.predict(X_test)
            pred1_train = model.predict(X_12)
            pred1_orig= model.predict(X1)
            model.update(X_12,y_12)
            pred2 = model.predict(X_test)
            pred2_orig= model.predict(X1)
            pred2_train =  model.predict(X_12)

            orig_mse[name].append(mean_squared_error(pred2_orig,y1))
            orig_stability[name].append(S1(pred1_orig,pred2_orig))
            orig_standard_stability[name].append(S2(pred1_orig,pred2_orig))

            train_mse[name].append(mean_squared_error(pred2_train,y_12))
            train_stability[name].append(S1(pred1_train,pred2_train))
            train_standard_stability[name].append(S2(pred1_train,pred2_train))
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(S1(pred1,pred2))
            standard_stability[name].append(S2(pred1,pred2))
        

    print(ds)
    for name in models.keys():
        print("="*80)
        print(f"{name}")
        orig_mse_scale = np.mean(orig_mse["baseline"]); orig_S1_scale = np.mean(orig_stability["baseline"]); orig_S2_scale = np.mean(orig_standard_stability["baseline"]);
        train_mse_scale = np.mean(train_mse["baseline"]); train_S1_scale = np.mean(train_stability["baseline"]); train_S2_scale = np.mean(train_standard_stability["baseline"]);
        mse_scale = np.mean(mse["baseline"]); S1_scale = np.mean(stability["baseline"]); S2_scale = np.mean(standard_stability["baseline"]);
        print(f"orig - mse: {np.mean(orig_mse[name]):.3f} ({np.mean(orig_mse[name])/orig_mse_scale:.2f}), stability: {np.mean(orig_stability[name]):.3f} ({np.mean(orig_stability[name])/orig_S1_scale:.2f}), standard stability: {np.mean(orig_standard_stability[name]):.3f} ({np.mean(orig_standard_stability[name])/orig_S2_scale:.2f})")
        print(f"train - mse: {np.mean(train_mse[name]):.3f} ({np.mean(train_mse[name])/train_mse_scale:.2f}), stability: {np.mean(train_stability[name]):.3f} ({np.mean(train_stability[name])/train_S1_scale:.2f}), standard stability: {np.mean(train_standard_stability[name]):.3f} ({np.mean(train_standard_stability[name])/train_S2_scale:.2f})")
        print(f"test - mse: {np.mean(mse[name]):.3f} ({np.mean(mse[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S1_scale:.2f}), standard stability: {np.mean(standard_stability[name]):.3f} ({np.mean(standard_stability[name])/S2_scale:.2f})")
        print("="*80)
    print()