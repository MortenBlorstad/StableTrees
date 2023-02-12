import numpy as np
import pandas as pd


# from examples in R package ISLR2, https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf
datasets =["Boston", "Carseats","College", "Hitters", "Wage"]
targets = ["medv", "Sales", "Apps", "Salary", "wage"]
drop_features = {dataset:[] for dataset in datasets}
drop_features["Wage"].append("logwage")
drop_features["Wage"].append("region")

ordinal_features = {dataset:[] for dataset in datasets}
ordinal_features["Carseats"].append("ShelveLoc")
ordinal_features["Wage"].append("education")

def handle_ordinal(dataset : pd.DataFrame,var : str) ->pd.DataFrame:
    print(var)
    if var == "ShelveLoc":
        
        dataset[var] = np.where(dataset[var]=="Bad", 0, dataset[var])
        dataset[var] = np.where(dataset[var]=="Medium", 1, 2)

    if var == "education":
        print("asdad")
        dataset[var] = dataset[var].astype(str).str[0].astype(int)

    return dataset
        
def data_preperation(dataname:str):
    data = pd.read_csv("data/"+ dataname+".csv") # load dataset
    data = data.dropna(axis=0, how="any") # remove missing values if any
    for var in drop_features[dataname]:
        data = data.drop(var, axis=1)

    for var in ordinal_features[dataname]:
        data = handle_ordinal(data,var)
    
    cat_data = data.select_dtypes("object") # find categorical features
    if not cat_data.empty: # if any categorical features, one-hot encode them
        cat_data = pd.get_dummies(data.select_dtypes("object"), prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
        data = pd.concat([data.select_dtypes(['int','float']),cat_data],axis=1)

    data = data.dropna(axis=0, how="any") # remove missing values if any
    return data



