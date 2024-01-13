
from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator 
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np
from node import Node
from splitters.splitter import Splitter

criterions = {"mse":0, "poisson":1}


class loss_function():
    def __init__(self,loss_type = "mse"):
        self.loss_type = loss_type
    
    def loss(self,y:np.ndarray, y_hat:np.ndarray):
        if self.loss_type =="mse":
            return (y - y_hat)**2
        raise Exception("loss type does not exist")
    
    def dloss(self,y:np.ndarray, y_hat:np.ndarray, y_prev:np.ndarray = None, gammas:np.ndarray = None):
        if self.loss_type =="mse":
            dloss = 2*(y_hat-y) 
            if y_prev is not None and gammas is not None:
                dloss+= 2*gammas*(y_prev-y)
            return dloss
        raise Exception("loss type does not exist")
    
    def ddloss(self,y:np.ndarray, y_hat:np.ndarray, y_prev:np.ndarray = None, gammas:np.ndarray = None):
        if self.loss_type =="mse":
            ddloss = np.ones(len(y))*2
            if y_prev is not None and gammas is not None:
                ddloss+= 2
            return ddloss
        raise Exception("loss type does not exist")
    
    def link_function(self,y):
        if self.loss_type =="mse":
            return y
        raise Exception("loss type does not exist")
    
    def inverse_link_function(self,y):
        if self.loss_type =="mse":
            return y
        raise Exception("loss type does not exist")


class BaseRegressionTree(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                  max_features:int = None, random_state : int = None) -> None:
        criterion = str(criterion).lower()
        if criterion not in criterions.keys():
            raise ValueError("Possible criterions are 'mse' and 'poisson'.")
        self.criterion = criterion

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)
        if max_features is None:
            max_features = 2147483647
        self.max_features = int(max_features)

        self.min_samples_split = float(min_samples_split)

        if random_state is None:
            random_state = 0
        self.random_state = int(random_state)

        self.adaptive_complexity = adaptive_complexity
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = 1


    def check_input(self,  X : np.ndarray ,y : np.ndarray):
        if X.ndim <2:
            X = np.atleast_2d(X)
        if np.issubdtype(X.dtype, np.number):
            X = X.astype("double")
        else:
            raise ValueError("X needs to be numeric")
        
        if y.ndim >1:
            raise ValueError("y needs to be 1-d")

        if np.issubdtype(y.dtype, np.number):
            y = y.astype("double")
        else:
            raise ValueError("y needs to be numeric")
        return X,y

    @abstractmethod
    def update(self,X : np.ndarray ,y : np.ndarray,sample_weight: np.ndarray = None):
        pass

    @abstractmethod
    def fit(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None): 
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)
    

    def plot(self):
        '''
        plots the tree. A visualisation of the tree
        '''
        plt.rcParams["figure.figsize"] = (20,10)
        self.__plot(self.root)
        plt.plot(0, 0, alpha=1) 
        plt.axis("off")
        

    def __plot(self,node: Node, x=0,y=-10,off_x = 100000,off_y = 15):
        '''
        a helper method to plot the tree. 
        '''
        

        # No child.
        if node is None:
            return

        if node.is_leaf():
            plt.plot(x+10, y-5, alpha=1) 
            plt.plot(x-10, y-5, alpha=1) 
            plt.text(x, y,f"{node.predict():.2f}", fontsize=8,ha='center') 
            plt.text(x, y-2,f"{node.nsamples():.2f}", fontsize=8,ha='center') 
            return
        
        
    
        x_left, y_left = x-off_x,y-off_y
        plt.text(x, y,f"X_{node.get_split_feature()}<={node.get_split_value():.4f}\n", fontsize=8,ha='center')
        plt.text(x, y-2,f"impurity: {node.get_impurity():.3f}", fontsize=8,ha='center')
        plt.text(x, y-4,f"nsamples: {node.get_features_indices()}", fontsize=8,ha='center')
        plt.annotate("", xy=(x_left, y_left+4), xytext=(x-2, y-4),
        arrowprops=dict(arrowstyle="->"))

        x_right, y_right = x+off_x,y-off_y
        plt.annotate("", xy=(x_right , y_right+4), xytext=(x+2, y-4),
        arrowprops=dict(arrowstyle="->"))
        self.__plot(node.get_left_node(),x_left, y_left, off_x*0.5)
        self.__plot(node.get_right_node() ,x_right, y_right,off_x*0.5)


class Tree(BaseRegressionTree):
    """
        Baseline: update method - same as the fit method. 
        Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.
    min_samples_leaf : int,  default = 5.
                Hyperparameter to determine the minimum number of samples required in a leaf node.
    adaptive_complexity : bool,  default = False.
                Hyperparameter to determine wheter find the tree complexity adaptively.
    max_features : bool,  default = None.
                The number of features to consider when looking for the best split.
    random_state : bool,  default = None.
                Controls the randomness of the tree.
    """

    def __init__(self, *,criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5,
                    adaptive_complexity : bool = False, max_features:int = None, random_state : int = None) -> None:
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf, adaptive_complexity,max_features, random_state)
        self.number_of_nodes = 0
        self.tree_depth = 0
        self.loss_function = loss_function(criterion)


    def all_same(self, y :np.ndarray ):
        return np.all(y == y[0])


    def build_tree(self,X : np.ndarray ,y : np.ndarray, g : np.ndarray, h : np.ndarray, depth:int, sample_weight: np.ndarray):
        self.number_of_nodes +=1
        self.tree_depth = max(depth,self.tree_depth)
        if X.shape[0]<2 or len(y)<2:
            return
        
        n = len(y)
        G = g.sum()
        H = h.sum()
        y_sum = (y*sample_weight).sum()
        sum_weights = sample_weight.sum()
        pred = self.loss_function.link_function(y_sum/sum_weights) - self.pred_0
        print(pred)
        any_split = None
        score  = np.inf
        
        split_value = None
        w_var = 1
        y_var = 1
        features_indices = np.arange(0, X.shape[1])

        loss_parent = ((y - pred)**2).sum()
        
        if self.all_same(y):
            return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        
        any_split, split_feature, split_value, score, y_var ,w_var,expected_max_S = self.splitter.find_best_split(X,y, g,h, features_indices)
        

        if depth >=self.max_depth:
           return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        if n< self.min_samples_split:
            return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        if not any_split:
            return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        
        left_mask = X[:,split_feature]<=split_value
        right_mask = ~left_mask
        node = Node(split_value, score, split_feature, n , pred, y_var, w_var,features_indices)
        node.left_child = self.build_tree(X[left_mask], y[left_mask] , g[left_mask] , h[left_mask], depth+1, sample_weight[left_mask])
        node.right_child = self.build_tree(X[right_mask], y[right_mask] , g[right_mask] , h[right_mask], depth+1, sample_weight[right_mask])

        if node.left_child!=None:
            node.left_child.w_var*=expected_max_S
            node.left_child.parent_expected_max_S=expected_max_S
        
        if node.right_child!=None:
            node.right_child.w_var*=expected_max_S
            node.right_child.parent_expected_max_S=expected_max_S

        return node
    
        

    def fit(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None): 
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.total_obs = len(y)
        self.splitter = Splitter(min_samples_leaf=self.min_samples_leaf, total_obs=self.total_obs , adaptive_complexity=self.adaptive_complexity, 
                max_features=None, learning_rate=1)
        
        self.pred_0 = self.loss_function.link_function(y.mean()) 
        pred = np.ones(self.total_obs)*self.pred_0
        
        g = self.loss_function.dloss(y, pred )*sample_weight 
        h = self.loss_function.ddloss(y, pred )*sample_weight
        self.root = self.build_tree(X, y, g, h, 0, sample_weight)
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = []
        for i in range(len(X)):
            obs = X[i, :]
            y_pred.append(self.predict_obs(obs))
        y_pred = np.array(y_pred)
        return self.loss_function.inverse_link_function(self.pred_0 + y_pred)
    
    def predict_obs(self, obs: np.ndarray) -> np.ndarray:
        node = self.root
        while node is not None:
            if node.is_leaf():
                return node.prediction
            else:
                if obs[node.split_feature] <= node.split_value:
                    node = node.left_child
                else:
                    node = node.right_child
        return None

    def update(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None):
        return self.fit(X,y,sample_weight)
    

class ABU(Tree):
    def __init__(self, *, criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, max_features: int = None, random_state: int = None) -> None:
        super().__init__(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, adaptive_complexity=adaptive_complexity, max_features=max_features, random_state=random_state)


    def update_tree(self,X : np.ndarray, y : np.ndarray, y_prev : np.ndarray, g : np.ndarray, h : np.ndarray,gammas:np.ndarray, depth:int, sample_weight: np.ndarray):
        self.number_of_nodes +=1
        self.tree_depth = max(depth,self.tree_depth)
        if X.shape[0]<2 or len(y)<2:
            return
        
        n = len(y)
        G = g.sum()
        H = h.sum()
        y_sum = (y*sample_weight).sum()
        sum_weights = sample_weight.sum()
        ypred1_sum = (y_prev * gammas).sum()
        sum_gammas = gammas.sum()
        pred = self.loss_function.link_function((y_sum+ypred1_sum)/((sum_weights+sum_gammas))) - self.pred_0;
        any_split = None
        score  = np.inf
        
        split_value = None
        w_var = 1
        y_var = 1
        features_indices = np.arange(0, X.shape[1])

        loss_parent = ((y - pred)**2).sum()
        
        if self.all_same(y):
            return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        
        any_split, split_feature, split_value, score, y_var ,w_var,expected_max_S = self.splitter.find_best_split(X,y, g,h, features_indices)
        

        if depth >=self.max_depth:
           return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        if n< self.min_samples_split:
            return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        if not any_split:
            return Node(split_value, score, None, n , pred, y_var, w_var,features_indices)
        
        left_mask = X[:,split_feature]<=split_value
        right_mask = ~left_mask
        node = Node(split_value, score, split_feature, n , pred, y_var, w_var,features_indices)
        node.left_child = self.build_tree(X[left_mask], y[left_mask] , g[left_mask] , h[left_mask], depth+1, sample_weight[left_mask])
        node.right_child = self.build_tree(X[right_mask], y[right_mask] , g[right_mask] , h[right_mask], depth+1, sample_weight[right_mask])

        if node.left_child!=None:
            node.left_child.w_var*=expected_max_S
            node.left_child.parent_expected_max_S=expected_max_S
        
        if node.right_child!=None:
            node.right_child.w_var*=expected_max_S
            node.right_child.parent_expected_max_S=expected_max_S

        return node
            
    def predict_info(self, X: np.ndarray) -> np.ndarray:
        info = np.zeros((X.shape[0],5))
        for i in range(len(X)):
            obs = X[i, :]
            info[i,:] = self.predict_info_obs(obs)
        return info
    
    def predict_info_obs(self, obs: np.ndarray) -> np.ndarray:
        node = self.root
        while node is not None:
            if node.is_leaf():
                return [node.prediction, node.y_var/node.w_var/node.num_samples,node.w_var, node.y_var,node.num_samples]
            else:
                if obs[node.split_feature] <= node.split_value:
                    node = node.left_child
                else:
                    node = node.right_child
        return None


    def update(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))

        info = self.predict_info(X)
        gammas = info[:,1]
        y_prev = info[:,0]
        self.total_obs = len(y)
        self.splitter = Splitter(min_samples_leaf=self.min_samples_leaf, total_obs=self.total_obs , adaptive_complexity=self.adaptive_complexity, 
                max_features=None, learning_rate=1)
        
        self.pred_0 = self.loss_function.link_function(y.mean()) 
        pred = np.ones(self.total_obs)*self.pred_0
        
        g = self.loss_function.dloss(y, pred, y_prev, gammas)*sample_weight 
        h = self.loss_function.ddloss(y, pred, y_prev, gammas )*sample_weight
        self.root = self.update_tree(X, y,y_prev, g, h,gammas, 0, sample_weight)


# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)

# n =1000
# x = np.random.uniform(0,4,(n,1))
# y = np.random.normal(x.ravel()**2,1,n)

# # from sklearn.datasets import load_boston
# # boston = load_boston()
# # y = boston.target
# # x = boston.data
# # X12,X_test,y12,y_test =  train_test_split(x, y, test_size=0.1, random_state=0)
# # X1,X2,y1,y2 =  train_test_split(X12, y12, test_size=0.5, random_state=0)

# X1 = np.random.uniform(0,4,(n,1))
# y1 = np.random.normal(X1.ravel()**2,1,n)

# X2 = np.random.uniform(0,4,(n,1))
# y2 = np.random.normal(X2.ravel()**2,1,n)

# X3 = np.random.uniform(0,4,(n,1))
# y3 = np.random.normal(X3.ravel()**2,1,n)


# X = np.vstack((X1,X2,X3))
# Y = np.concatenate((y1,y2,y3))

# X_test = np.sort(np.random.uniform(0,4,(n,1)),axis=0)
# y_test = np.random.normal(X_test.ravel()**2,1,n)

# pca.fit(X1)

# t1 = Tree(adaptive_complexity=True,min_samples_leaf=5)
# t = ABU(adaptive_complexity=True,min_samples_leaf=5)

# t.fit(X1,y1)
# t1.fit(X,Y)
# pred = t1.predict(X_test)
# # t.plot()
# # plt.show()
# X2_ = np.vstack((X1,X2))
# Y2_ = np.concatenate((y1,y2))
# t.update(X2_,Y2_)
# X3_ = np.vstack((X2_,X3))
# Y3_ = np.concatenate((Y2_,y3))
# t.update(X3_,Y3_)
# pred2 = t.predict(X_test)

# import matplotlib.pyplot as plt

# fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)#
# print(X_test.shape, y_test.shape)

# # ax1.scatter(pca.transform(X_test)[:,0],y_test,c = "red")
# # ax1.scatter(pca.transform(X_test)[:,0],pred,c = "black")
# # ax2.scatter(pca.transform(X_test)[:,0],y_test,c = "red")
# # ax2.scatter(pca.transform(X_test)[:,0],pred2,c = "black")

# ax1.scatter(X_test,y_test,c = "red")
# ax1.plot(X_test,pred,c = "black")
# ax2.scatter(X_test,y_test,c = "red")
# ax2.plot(X_test,pred2,c = "black")
# plt.show()