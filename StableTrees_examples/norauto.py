import pandas as pd
import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from stabletrees import BaseLineTree,AbuTreeI,NaiveUpdate,TreeReevaluation,StabilityRegularization, BABUTree
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.linear_model import PoissonRegressor
EPSILON = 1.1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)
SEED = 0
plt.rcParams["figure.figsize"] = (20,12)



with tarfile.open("data\poisson\\norauto.tar.gz", "r:*") as tar:
    csv_path = tar.getnames()[0]
    df = pd.read_csv(tar.extractfile(csv_path), header=0)

    
df["DistLimit_ints"] = pd.Categorical(df.DistLimit, categories=["8000 km", "12000 km", "16000 km", "20000 km","25000-30000 km", "no limit"], ordered=True).rename_categories(np.arange(6))


df["GeoRegion_ints"] = pd.Categorical(df.GeoRegion, categories=['Low-','Low+','Medium-', 'Medium+', 'High-','High+'], ordered=True).rename_categories(np.arange(6))


df["y"] = df.NbClaim/df.Expo

df.insert(len(df.columns)-1, 'y', df.pop('y'))

ClaimNb = df.NbClaim.to_numpy()
Exposure = df.Expo.to_numpy()
df.drop(["DistLimit","GeoRegion","ClaimAmount","Expo","NbClaim"],axis=1,inplace=True)


X = df.to_numpy()[:,:-1]
y = df.to_numpy()[:,-1]

t = StabilityRegularization(criterion = "poisson", adaptive_complexity=True,lmbda=0.75).fit(X,y)
t.update(X,y)
print(mean_poisson_deviance(ClaimNb,t.predict(X)*Exposure))

models = {  
            "baseline": BaseLineTree(),
            "poisReg": PoissonRegressor(alpha=0.001),
            "NU": NaiveUpdate(),
            "TR":TreeReevaluation(delta=0.1),
            "SR":StabilityRegularization(),
            "ABU":AbuTreeI(),
            "BABU": BABUTree()
            }
stability_all = {name:[] for name in models.keys()}
standard_stability_all= {name:[] for name in models.keys()}
mse_all= {name:[] for name in models.keys()}

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

print(np.max(y))

for train_index, test_index in kf.split(X):
    X_12, y_12 = X[train_index],y[train_index]
    ClaimNb_12 = ClaimNb[train_index]
    Exposure_12  = Exposure[train_index]
    X_test,y_test = X[test_index],y[test_index]
    ClaimNb_test = ClaimNb[test_index]
    Exposure_test  = Exposure[test_index]
    X1,X2,y1,y2,ClaimNb_1,ClaimNb_2,Exposure_1,Exposure_2 =  train_test_split(X_12, y_12,ClaimNb_12,Exposure_12, test_size=0.5, random_state=SEED)
   
    # initial model 
    criterion = "mse"
    models = {  
            "baseline": BaseLineTree(criterion = criterion, adaptive_complexity=True,min_samples_leaf=20),
            "poisReg": PoissonRegressor(alpha=0.001),
            #"NU": NaiveUpdate(criterion = criterion, adaptive_complexity=True),
            #"TR":TreeReevaluation(criterion = criterion, adaptive_complexity=True, delta=0.1),
            "SR":StabilityRegularization(criterion = criterion, adaptive_complexity=True,lmbda=0.75,min_samples_leaf=20)
            #"ABU":AbuTreeI(criterion = criterion, adaptive_complexity=True),
            #"BABU": BABUTree(criterion = criterion, adaptive_complexity=True),

            }
    for name, model in models.items():
        model.fit(X1,y1)
        
        pred1 = model.predict(X_test)
        pred1_train = model.predict(X_12)
        pred1_orig= model.predict(X1)
        #print("before")
        
        if name == "poisReg":
            model.fit(X_12,y_12)
        else:
            model.update(X_12,y_12)
        #print("after")
        pred2 = model.predict(X_test)
        pred2_orig= model.predict(X1)
        pred2_train =  model.predict(X_12)
        
        if np.any(np.isnan(pred2_orig)  )or np.any(np.isinf(pred2_orig)  ):
            plt.hist(y1)
            plt.show()
            plt.hist(pred2_orig[~np.isinf(pred2_orig)])
            plt.show()
            models["SR"].plot()
            plt.show()
        if name == "SR":
            print(mean_poisson_deviance(ClaimNb_1+0.00001,pred2_orig*Exposure_1+0.00001))
        orig_mse[name].append(mean_poisson_deviance(ClaimNb_1+0.00001,pred2_orig*Exposure_1+0.00001))
        orig_stability[name].append(S1(pred1_orig*Exposure_1,pred2_orig*Exposure_1))
        orig_standard_stability[name].append(S2(pred1_orig*Exposure_1,pred2_orig*Exposure_1))

        train_mse[name].append(mean_poisson_deviance(ClaimNb_12+0.00001,pred2_train*Exposure_12+0.00001))
        train_stability[name].append(S1(pred1_train*Exposure_12,pred2_train*Exposure_12))
        train_standard_stability[name].append(S2(pred1_train*Exposure_12,pred2_train*Exposure_12))
        mse[name].append(mean_poisson_deviance(ClaimNb_test+0.00001,pred2*Exposure_test+0.00001))
        stability[name].append(S1(pred1*Exposure_test,pred2*Exposure_test))
        standard_stability[name].append(S2(pred1*Exposure_test,pred2*Exposure_test))
    
for name in models.keys():
    print("="*80)
    print(f"{name}")
    orig_mse_scale = np.mean(orig_mse["baseline"]); orig_S1_scale = np.mean(orig_stability["baseline"]); orig_S2_scale = np.mean(orig_standard_stability["baseline"]);
    train_mse_scale = np.mean(train_mse["baseline"]); train_S1_scale = np.mean(train_stability["baseline"]); train_S2_scale = np.mean(train_standard_stability["baseline"]);
    mse_scale = np.mean(mse["baseline"]); S1_scale = np.mean(stability["baseline"]); S2_scale = np.mean(standard_stability["baseline"]);
    
    print(f"orig - poisson: {np.mean(orig_mse[name]):.3f} ({np.mean(orig_mse[name])/orig_mse_scale:.2f}), stability: {np.mean(orig_stability[name]):.3f} ({np.mean(orig_stability[name])/orig_S1_scale:.2f}), standard stability: {np.mean(orig_standard_stability[name]):.3f} ({np.mean(orig_standard_stability[name])/orig_S2_scale:.2f})")
    print(f"train - poisson: {np.mean(train_mse[name]):.3f} ({np.mean(train_mse[name])/train_mse_scale:.2f}), stability: {np.mean(train_stability[name]):.3f} ({np.mean(train_stability[name])/train_S1_scale:.2f}), standard stability: {np.mean(train_standard_stability[name]):.3f} ({np.mean(train_standard_stability[name])/train_S2_scale:.2f})")
    print(f"test - poisson: {np.mean(mse[name]):.3f} ({np.mean(mse[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S1_scale:.2f}), standard stability: {np.mean(standard_stability[name]):.3f} ({np.mean(standard_stability[name])/S2_scale:.2f})")
    print("="*80)
    mse_all[name] += [score/mse_scale for score in mse[name]]
    stability_all[name] += [score/S1_scale for score in stability[name]]
    standard_stability_all[name] += [score/S2_scale for score in standard_stability[name]]
print()

