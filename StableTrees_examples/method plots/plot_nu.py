from sklearn.datasets import load_diabetes
from stabletrees import TreeReevaluation
import numpy as np
from matplotlib import pyplot as plt
from plotter import plot
from sklearn.model_selection import train_test_split

X1 = np.array([4,5,6]).reshape(-1,1)
y1 = np.array([0,1,4])
X12 = np.array([4,5,6,6]).reshape(-1,1)
y12 = np.array([0,1,4,5])
X,y = load_diabetes(return_X_y=True)

X1,X2, y1, y2 = train_test_split(X,y,test_size=0.2,random_state=0)

t = TreeReevaluation(max_depth=2,delta=0.05).fit(X1,y1)

t.update(X,y)

plot(t.root)
plt.show()

