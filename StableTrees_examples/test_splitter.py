import numpy as np
X = np.array([4,5,6]).reshape(-1,1)
y = np.array([0,1,4])

from stabletrees import BaseLineTree

tree = BaseLineTree(max_depth = 1)

tree.fit(X,y)