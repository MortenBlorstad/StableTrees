from matplotlib import pyplot as plt
import os

cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.append(cur_file_path + '\\..\\cpp\\build\\Release\\')
from stable_trees import Node




def plot(root: Node):
    '''
    plots the tree. A visualisation of the tree
    '''
    plt.rcParams["figure.figsize"] = (20,10)
    __plot(root)
    plt.plot(0, 0, alpha=1) 
    plt.axis("off")
    plt.show()
    
def __plot(node: Node,x=0,y=-10,off_x = 100000,off_y = 15):
    '''
    a helper method to plot the tree. 
    '''
    

    # No child.
    if node is None:
        return

    if node.is_leaf():
        plt.plot(x+10, y-5, alpha=1) 
        plt.plot(x-10, y-5, alpha=1) 
        plt.text(x, y,f"{node.text()}", fontsize=8,ha='center') 
        return
    
    
  
    x_left, y_left = x-off_x,y-off_y
    plt.text(x, y,f"{node.text()}", fontsize=8,ha='center')
    plt.text(x, y-2,f"impurity: {node.get_split_score():.3f}", fontsize=8,ha='center')
    plt.text(x, y-4,f"nsamples: {node.nsamples()}", fontsize=8,ha='center')
    plt.annotate("", xy=(x_left, y_left+4), xytext=(x-2, y-4),
    arrowprops=dict(arrowstyle="->"))

    x_right, y_right = x+off_x,y-off_y
    plt.annotate("", xy=(x_right , y_right+4), xytext=(x+2, y-4),
    arrowprops=dict(arrowstyle="->"))
    __plot(node.get_left_node(),x_left, y_left, off_x*0.5)
    __plot(node.get_right_node() ,x_right, y_right,off_x*0.5)

    return 

if __name__ == "__main__":
    from stable_trees import Tree
    import time
    from sklearn import datasets
    from sklearn.tree import DecisionTreeRegressor,plot_tree
    import numpy as np  
    #X  = np.array([[1,1],[1,2],[2,4],[2,3],[4,5]])
    #y = X[:,0]*2 + X[:,1]*0.5 
    X,y= datasets.make_regression(100000,10, random_state=0)
    start = time.time()
    clf = DecisionTreeRegressor(random_state=0)
    clf = clf.fit(X,y)
    clf.predict(X)
    end = time.time()
    print("sklearn: ",end - start)
    
    start = time.time()
    tree = Tree()
    tree.learn(X,y)
    tree.predict(X)
    end = time.time()
    print("my impl: ",end - start)
    #treebuilder.example()
    #root = tree.get_root()
    #print(tree.predict(X),y))
    #plot(root)
    
    
   
            