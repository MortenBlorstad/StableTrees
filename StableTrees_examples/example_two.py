from stabletrees import BaseLineTree, StableTree1,StableTree2
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np

from sklearn.datasets import make_regression

SEED = 0
EPSILON = 1.1
parameters = {"criterion": ["poisson"], 'max_depth':[None, 5, 10], 'min_samples_split':[2,4,8]}

clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)
import numpy as np
np.random.seed(SEED)
X = np.random.multivariate_normal([0.025 ,0.075,0.05], np.array([[1, 0.1, 0], [0.1,1, 0.2], [0,0.2,1]]), size=1000)
def formula(X, noise = 0.1):
    return  np.exp(2*X[:,0] + 0.1*X[:,1] + 0.75*X[:,2] + np.random.normal(0,noise))
y = formula(X)


kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)

models = {  
                 "baseline": BaseLineTree(),
                 "method1" : StableTree1(),
                 "method2" : StableTree2()
            }


print(X.shape,y.shape)

stability = {name:[] for name in models.keys()}
standard_stability = {name:[] for name in models.keys()}
poisson = {name:[] for name in models.keys()}
train_stability = {name:[] for name in models.keys()}
train_standard_stability = {name:[] for name in models.keys()}
train_poisson = {name:[] for name in models.keys()}
orig_stability = {name:[] for name in models.keys()}
orig_standard_stability = {name:[] for name in models.keys()}
orig_poisson= {name:[] for name in models.keys()}

for train_index, test_index in kf.split(X):
    X_12, y_12 = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
    clf.fit(X1,y1)
    params = clf.best_params_
    # initial model 
    models = {  
                "baseline": BaseLineTree(**params),
                "method1" : StableTree1(**params, delta = 0.25),
                "method2" : StableTree2(**params)
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

        orig_poisson[name].append(mean_poisson_deviance(pred2_orig,y1))
        orig_stability[name].append(np.std(np.log((pred1_orig+EPSILON)/(pred2_orig+EPSILON))))
        orig_standard_stability[name].append(np.mean(abs(pred1_orig- pred2_orig)))

        train_poisson[name].append(mean_poisson_deviance(pred2_train,y_12))
        train_stability[name].append(np.std(np.log((pred1_train+EPSILON)/(pred2_train+EPSILON))))
        train_standard_stability[name].append(np.mean(abs(pred1_train- pred2_train)))
        poisson[name].append(mean_poisson_deviance(y_test,pred2))
        stability[name].append(np.std(np.log((pred1+EPSILON)/(pred2+EPSILON))))
        standard_stability[name].append(np.mean(abs(pred1- pred2)))
    


for name in models.keys():
    print("="*80)
    print(f"{name}")
    orig_mse_scale = np.mean(orig_poisson["baseline"]); orig_S1_scale = np.mean(orig_stability["baseline"]); orig_S2_scale = np.mean(orig_standard_stability["baseline"]);
    train_mse_scale = np.mean(train_poisson["baseline"]); train_S1_scale = np.mean(train_stability["baseline"]); train_S2_scale = np.mean(train_standard_stability["baseline"]);
    mse_scale = np.mean(poisson["baseline"]); S1_scale = np.mean(stability["baseline"]); S2_scale = np.mean(standard_stability["baseline"]);
    print(f"orig - poisson dev: {np.mean(orig_poisson[name]):.3f} ({np.mean(orig_poisson[name])/orig_mse_scale:.2f}), stability: {np.mean(orig_stability[name]):.3f} ({np.mean(orig_stability[name])/orig_S1_scale:.2f}), standard stability: {np.mean(orig_standard_stability[name]):.3f} ({np.mean(orig_standard_stability[name])/orig_S2_scale:.2f})")
    print(f"train - poisson dev: {np.mean(train_poisson[name]):.3f} ({np.mean(train_poisson[name])/train_mse_scale:.2f}), stability: {np.mean(train_stability[name]):.3f} ({np.mean(train_stability[name])/train_S1_scale:.2f}), standard stability: {np.mean(train_standard_stability[name]):.3f} ({np.mean(train_standard_stability[name])/train_S2_scale:.2f})")
    print(f"test - poisson dev: {np.mean(poisson[name]):.3f} ({np.mean(poisson[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S1_scale:.2f}), standard stability: {np.mean(standard_stability[name]):.3f} ({np.mean(standard_stability[name])/S2_scale:.2f})")
    print("="*80)
print()
