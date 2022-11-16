'''
    Insurance claims frequency simulation using Poisson process
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
explanatory_vars = {"obj_age":0.3, "obj_size":0.07, "obj_type": {1:0.01, 2:0.08, 3:0.025}}
np.random.seed(0)


def formula(X):
    return 0.07*X.iloc[:,1]  + 0.00009*X.iloc[:,2] + 0.0007*X.iloc[:,3] + 0.008*X.iloc[:,4] + 0.0025*X.iloc[:,5] + 0.05

def create_policies(number_of_policiec:int) ->pd.DataFrame:
    X = {"obj_id" : 100000000 + np.arange(1,number_of_policiec+1),
        "obj_age": np.random.randint(0,20,size = number_of_policiec),
        "obj_size": np.random.randint(1000,2000,size = number_of_policiec),
        "obj_type": np.random.choice(["1","2","3"], p=[0.25,0.5,0.25],size = number_of_policiec)}

    X = pd.DataFrame(X)
    X = pd.get_dummies(X,"obj_type")
    y = pd.DataFrame(formula(X), columns=["y"])
    return pd.concat([X,y], axis=1)

def simulate_claim_frequencies(number_of_policiec:int) -> pd.DataFrame:
    data = create_policies(number_of_policiec)
    freq = np.random.poisson(data.y)

    obs = {"obj_id":[], "sev":[]}
    for i,f in enumerate(freq):
        id = data.obj_id[i]
        if f>1:
            sev = np.round(np.random.lognormal(size = f, mean=9,sigma=1.75),2)
            obs["obj_id"] += [id]*len(sev)
            obs["sev"] += sev.tolist()
        else:
            obs["sev"].append(0)
            obs["obj_id"].append(id)
    

    return pd.DataFrame(obs).merge(data, on="obj_id",how='left')    

if __name__ == '__main__':
    from scipy.stats import poisson
    N = 1000
    data = simulate_claim_frequencies(N)

    (data.groupby("obj_id").sev.count()-1).hist(density=True)

    # creating a numpy array for x-axis
    x = np.arange(0, 5, 0.1)
    # poisson distribution data for y-axis
    y = poisson.pmf(x, mu=0.5)
    
    # plotting the graph
    plt.plot(x, y)

    plt.show()





