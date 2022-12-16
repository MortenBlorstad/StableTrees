from stabletrees import BaseLineTree, StableTree
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np

SEED = 0
EPSILON = 1.1
X,y = make_regression(n_samples=1000, n_features=10, n_informative=5,noise=10,random_state=0)
y = y + np.max(np.abs(y))

parameters = {'max_depth':[None, 5, 10], 'min_samples_split':[2,4,8]}
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)
clf.fit(X,y)
params = clf.best_params_
print(params)
models = {  
                "baseline": BaseLineTree(**params),
                "method2":StableTree(**params)
        }
stability = {name:[] for name in models.keys()}
standard_stability = {name:[] for name in models.keys()}
mse = {name:[] for name in models.keys()}
train_stability = {name:[] for name in models.keys()}
train_standard_stability = {name:[] for name in models.keys()}
train_mse = {name:[] for name in models.keys()}
iteration = 1
kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)


for train_index, test_index in kf.split(X):
    X_12, y_12 = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)

    
    # initial model 
    
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
    
        if (iteration) % 10 ==0:
            print(f"{iteration}/{50}, {name}")
            print(f"train - mse: {np.mean(train_mse[name]):.3f}, stability: {np.mean(train_stability[name]):.3f}, standard stability: {np.mean(train_standard_stability[name]):.3f}")
            print(f"test - mse: {np.mean(mse[name]):.3f}, stability: {np.mean(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")
    iteration+=1

print(models)
for name in models.keys():
    print("="*80)
    print(f"{name}")
    print(f"train - mse: {np.mean(train_mse[name]):.3f}, stability: {np.mean(train_stability[name]):.3f}, standard stability: {np.mean(train_standard_stability[name]):.3f}")
    print(f"test - mse: {np.mean(mse[name]):.3f}, stability: {np.mean(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")
    print("="*80)

