import time
from sklearn import datasets
import numpy as np  

from scipy.stats import skew, kurtosis


X,y= datasets.make_regression(2000, 10, random_state=0)


def get_state(X,y):
    skews = skew(X,axis = 0)
    kurts = kurtosis(X,axis = 0)
    cors = np.apply_along_axis(lambda x: np.corrcoef(x,y)[0,1],0, X)
    ranges = np.ptp(X,axis=0)
    y_range = np.array([np.max(y)])
    state = np.concatenate((skews,kurts,cors,ranges,y_range))
    print(state.shape)
    return state


print(get_state(X,y))
mask = X[:,0]>np.mean(X[:,0])
print(get_state(X[mask],y[mask]))
print(get_state(X[~mask],y[~mask]))





