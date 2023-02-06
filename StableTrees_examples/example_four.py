from stabletrees import BaseLineTree,StableTree0, StableTree1,StableTree2,AbuTreeI
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SEED = 0
EPSILON = 1.1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))

def S2(pred1, pred2):
    return np.mean(abs(pred1- pred2))

parameters = {'max_depth':[None, 5, 10], 'min_samples_split':[2,4,8]}
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)

from sklearn.datasets import load_diabetes,make_regression


iteration = 1

#X,y = load_diabetes(return_X_y=True)
for i in range(2):
    np.random.seed(i)
    n = 500
    X =np.random.uniform(size=(n,1), low = 10,high = 20)
    y =np.random.normal(loc = X.ravel(),scale= 1, size=n)
    print(y.mean())
    #X,y= make_regression(2000,10, random_state=SEED, noise=10)
    # y = y + np.abs(np.min(y))
    # print(np.min(y),np.max(y))

    SEED = 0
    EPSILON = 1.1


    clf.fit(X,y)
    params = clf.best_params_
    print(params)
    # initial model 
    models = {  
                    "baseline w/ info crit": BaseLineTree(adaptive_complexity=True),
                    "baseline w/ hyperparam search": BaseLineTree(),
                    "method0": StableTree0(),
                    "method1":StableTree1(delta=0.05),
                    "method2":StableTree2(lmda=0.75),
                    "method3": AbuTreeI()
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
    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)

    stability_all = {name:[] for name in models.keys()}
    standard_stability_all= {name:[] for name in models.keys()}
    mse_all= {name:[] for name in models.keys()}

    for train_index, test_index in kf.split(X):
        # X_12, y_12 = X[train_index],y[train_index]
        # X_test,y_test = X[test_index],y[test_index]
        # X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
        X1 =np.random.uniform(size=(n,1), low = 10,high = 20)
        y1 =np.random.normal(loc = X1.ravel(),scale= 1, size=n)
        X2 =np.random.uniform(size=(n,1), low = 10,high = 20)
        y2 =np.random.normal(loc = X2.ravel(),scale= 1, size=n)
        X_test =np.random.uniform(size=(n,1), low = 10,high = 20)
        y_test =np.random.normal(loc = X_test.ravel(),scale= 1, size=n)
        X12 = np.vstack((X1,X2))
        y12 = np.concatenate((y1,y2))
        clf.fit(X1,y1)
        params = clf.best_params_
        # initial model 
        models = {  
                    "baseline w/ info crit": BaseLineTree(adaptive_complexity=True),
                    "baseline w/ hyperparam search": BaseLineTree(**params),
                    "method0": StableTree0(**params),
                    #"method1": StableTree1(**params, delta=0.05),
                    #"method2": StableTree2(**params,lmda=0.75),
                    #"method3": AbuTreeI()
                }
                
        for name, model in models.items():
            model.fit(X1,y1)
            
            pred1 = model.predict(X_test)
            pred1_train = model.predict(X12)
            pred1_orig= model.predict(X1)
            model.update(X12,y12)
            
            pred2 = model.predict(X_test)

            pred2_orig= model.predict(X1)
            pred2_train =  model.predict(X12)
            orig_mse[name].append(mean_squared_error(pred2_orig,y1))
            orig_stability[name].append(S1(pred1_orig,pred2_orig))
            orig_standard_stability[name].append(S2(pred1_orig,pred2_orig))

            train_mse[name].append(mean_squared_error(pred2_train,y12))
            train_stability[name].append(S1(pred1_train,pred2_train))
            train_standard_stability[name].append(S2(pred1_train,pred2_train))
            
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(S1(pred1,pred2))
            standard_stability[name].append(S2(pred1,pred2))
            


    for name in models.keys():
        print("="*80)
        print(f"{name}")
        orig_mse_scale = np.mean(orig_mse["baseline w/ info crit"]); orig_S1_scale = np.mean(orig_stability["baseline w/ info crit"]); orig_S2_scale = np.mean(orig_standard_stability["baseline w/ info crit"]);
        train_mse_scale = np.mean(train_mse["baseline w/ info crit"]); train_S1_scale = np.mean(train_stability["baseline w/ info crit"]); train_S2_scale = np.mean(train_standard_stability["baseline w/ info crit"]);
        mse_scale = np.mean(mse["baseline w/ info crit"]); S1_scale = np.mean(stability["baseline w/ info crit"]); S2_scale = np.mean(standard_stability["baseline w/ info crit"]);
        print(f"dataset 1 - mse: {np.mean(orig_mse[name]):.3f} ({np.mean(orig_mse[name])/orig_mse_scale:.2f}), stability: {np.mean(orig_stability[name]):.3f} ({np.mean(orig_stability[name])/orig_S1_scale:.2f}), standard stability: {np.mean(orig_standard_stability[name]):.3f} ({np.mean(orig_standard_stability[name])/orig_S2_scale:.2f})")
        print(f"dataset 1&2 - mse: {np.mean(train_mse[name]):.3f} ({np.mean(train_mse[name])/train_mse_scale:.2f}), stability: {np.mean(train_stability[name]):.3f} ({np.mean(train_stability[name])/train_S1_scale:.2f}), standard stability: {np.mean(train_standard_stability[name]):.3f} ({np.mean(train_standard_stability[name])/train_S2_scale:.2f})")
        print(f"test data - mse: {np.mean(mse[name]):.3f} ({np.mean(mse[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S1_scale:.2f}), standard stability: {np.mean(standard_stability[name]):.3f} ({np.mean(standard_stability[name])/S2_scale:.2f})")
        print("="*80)
        mse_all[name] += [score/mse_scale for score in mse[name]]
        stability_all[name] += [score/S1_scale for score in stability[name]]
        standard_stability_all[name] += [score/S2_scale for score in standard_stability[name]]

print(len(mse_all['baseline w/ info crit']))


plt.rcParams["figure.figsize"] = (16,8)


labels, data = mse_all.keys(), mse_all.values()
plt.boxplot(data)
plt.ylabel('mse',fontsize=12)
plt.xlabel("models")
plt.xticks(range(1, len(labels) + 1), labels)
plt.show()