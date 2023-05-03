import numpy as np
import pandas as pd
from stabletrees import AbuTree,BABUTree,BaseLineTree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
EPSILON = 0.000001

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)

np.random.seed(0)
n = 1000 # data size
B = 1000 # number of simulations
loss_d1 = []
loss_d2 = []
loss_d12 = []
loss_abu = []

stability_base = []
stability_abu = []
plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}


for b in range(B):
    X = np.random.uniform(size=(n,1),low=0,high=4)
    y = np.random.normal(loc=X.ravel(),scale=1,size = n)
    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=b)
    tree1 = AbuTree(criterion="mse", adaptive_complexity=True).fit(X1,y1) 
    tree2 = AbuTree(criterion="mse", adaptive_complexity=True).fit(X2,y2)
    tree3 = AbuTree(criterion="mse", adaptive_complexity=True).fit(X,y)
    tree4 = AbuTree(criterion="mse", adaptive_complexity=True).fit(X1,y1)
    tree4.update(X2,y2)
    stability_base.append(S2(tree1.predict(X),tree3.predict(X)))
    stability_abu.append(S2(tree1.predict(X),tree4.predict(X)))
    loss_d1.append(mean_squared_error(y,tree1.predict(X)))
    loss_d2.append(mean_squared_error(y,tree2.predict(X)))
    loss_d12.append(mean_squared_error(y,tree3.predict(X)))
    loss_abu.append(mean_squared_error(y,tree4.predict(X)))

fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True,dpi=500)
plt.rcParams.update(plot_params)
sns.kdeplot(loss_d1,label=r'$P(w|\mathcal{D}_1)$')
sns.kdeplot(loss_d2,label=r'$P(w|\mathcal{D}_2)$')
sns.kdeplot(loss_d12,label=r'$P(w|\mathcal{D}_1 \cup \mathcal{D}_2)$')
sns.kdeplot(loss_abu,label='ABU')
plt.xlabel("mse",fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\ABU_experiment.png",bbox_inches='tight')
plt.close()