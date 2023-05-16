from sklearn.datasets import load_diabetes
from stabletrees import NaiveUpdate
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from plote

X1 = np.array([4,5,6]).reshape(-1,1)
y1 = np.array([0,1,4])
X12 = np.array([4,5,6,6]).reshape(-1,1)
y12 = np.array([0,1,4,5])
X,y = load_diabetes(return_X_y=True)

X1,X2, y1, y2 = train_test_split(X,y,test_size=0.3,random_state=0)

t = NaiveUpdate(max_depth=2).fit(X1,y1)
#plt.subplot(1,2,1)
plot(t.root)
plt.show()
t.update(X,y)
plt.subplot(1,2,2)

