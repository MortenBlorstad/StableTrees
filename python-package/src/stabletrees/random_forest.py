
from stabletrees.tree import BaseRegressionTree,BaseLineTree,NaiveUpdate,AbuTree,TreeReevaluation,StabilityRegularization,BABUTree
from _stabletrees import RandomForest as rf
from _stabletrees import RandomForestSL as rfsl
from _stabletrees import RandomForestNU as rfnu
from _stabletrees import RandomForestTR as rftr
from _stabletrees import RandomForestABU as rfabu
from _stabletrees import RandomForestBABU as rfbabu
from _stabletrees import StackedRandomForest as stackedrf


import numpy as np

max_features_to_int = {"all": lambda x :x,
                       "third": lambda x :max(1, int(x/3)),
                       None : lambda x :x}

method_to_int = {"base":0,
                 "nu":1,
                 "tr":2,
                 "sl":3,
                 "abu":4,
                 "babu":5}
criterions = {"mse":0, "poisson":1}




class RandomForest(BaseRegressionTree):
    def __init__(self,n_estimators:int,max_features:str = "all", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        assert n_estimators>=1
        self.n_estimators = n_estimators
        self.max_features = max_features
        if self.max_features not in max_features_to_int.keys():
            self.max_features = "all"
        self.forest = []
       


    def fit(self,X,y):
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        self.forest = []
        for b in range(self.n_estimators):
            ind_b = np.random.choice(np.arange(0,n,1,dtype=int),replace=True, size =n )
            #features_b = np.random.choice(np.arange(0,num_max_features,1),replace=False, size =num_max_features).astype(int)
            X_b = X[ind_b]
            y_b = y[ind_b]
            t = BaseLineTree(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity, random_state = self.random_state).fit(X_b,y_b)
            self.forest.append(t)
        return self
        
    def update(self, X: np.ndarray, y: np.ndarray):
        return self.fit(X,y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert len(self.forest)>=1
        y_pred = np.zeros(X.shape[0])
        for t in self.forest:
            y_pred+= t.predict(X)

        return y_pred/self.n_estimators
    

class StackedRF(BaseRegressionTree):
    def __init__(self,method:str = "base",n_estimators:int = 100,max_features:str = "all", criterion: str = "mse", max_depth: int = None,
                  min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None, gamma:float = 0.5, learning_rate:float = 0.001) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.method = method
        self.criterion =criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        if self.max_features not in max_features_to_int.keys():
            self.max_features = "all"

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)
        self.forest = None
        self.learning_rate= learning_rate

    def fit(self,X,y,sample_weight=None):
        max_feature = max_features_to_int[self.max_features](X.shape[1])
        self.forest = stackedrf(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.gamma,self.learning_rate)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.forest.learn(X,y,sample_weight)
        return self
    
    def update(self, X: np.ndarray, y: np.ndarray,sample_weight=None):
        if self.forest is None:
            return self.fit(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.forest.update(X,y,sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert (self.forest is not None)
        return self.forest.predict(X)


class RF(BaseRegressionTree):
    def __init__(self,method:str = "base",n_estimators:int = 100,max_features:str = "all", criterion: str = "mse", max_depth: int = None,
                  min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None, gamma:float = 0.5, delta:float = 0.05,alpha:float = 0.0, bumping_iterations =5) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.bumping_iterations = bumping_iterations
        assert n_estimators>=1
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.method = method
        self.criterion =criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        if self.max_features not in max_features_to_int.keys():
            self.max_features = "all"

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)
        self.forest = None


    
        

    def fit(self,X,y,sample_weight=None):
        max_feature = max_features_to_int[self.max_features](X.shape[1])
        #print(method_to_int[self.method])
        if method_to_int[self.method] ==0:
            self.forest = rf(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature)
        if method_to_int[self.method] ==1:
            self.forest = rfnu(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature)
        if method_to_int[self.method] ==2:
            self.forest = rftr(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.delta,self.alpha)
        if method_to_int[self.method] ==3:
            self.forest = rfsl(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.gamma)
        if method_to_int[self.method] ==4:
            self.forest = rfabu(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature)
        if method_to_int[self.method] ==5:
            self.forest = rfbabu(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.bumping_iterations)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.forest.learn(X,y,sample_weight)
        return self
        
    def update(self, X: np.ndarray, y: np.ndarray,sample_weight=None):
        
        if self.forest is None:
            return self.fit(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.forest.update(X,y,sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert (self.forest is not None)
        return self.forest.predict(X)
        


class NaiveRandomForest(RandomForest):
    def __init__(self,n_estimators:int,max_features:str = "all", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:

        super().__init__(n_estimators,max_features,criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        
       


    def fit(self,X,y):
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        self.forest = []
        for b in range(self.n_estimators):
            ind_b = np.random.choice(np.arange(0,n,1,dtype=int),replace=True, size =n )
            #features_b = np.random.choice(np.arange(0,num_max_features,1),replace=False, size =num_max_features).astype(int)
            X_b = X[ind_b]
            y_b = y[ind_b]
            t = NaiveUpdate(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity, random_state = self.random_state).fit(X_b,y_b)
            self.forest.append(t)
        return self
    
    def update(self, X: np.ndarray, y: np.ndarray):
        assert len(self.forest)>=1
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        for i,t in enumerate(self.forest):
            ind_b = np.random.choice(np.arange(0,n,1,dtype=int),replace=True, size =n )
            X_b = X[ind_b]
            y_b = y[ind_b]
            self.forest[i].update(X_b,y_b)
        return self




class AbuRandomForest(RandomForest):
    def __init__(self,n_estimators:int,max_features:str = "all", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:

        super().__init__(n_estimators,max_features,criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        
    def fit(self,X,y):
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        self.forest = []
        self.indices = np.zeros((n,self.n_estimators),dtype=int)
        for b in range(self.n_estimators):
            ind_b = np.random.choice(np.arange(0,n,1,dtype=int),replace=True, size =n )
            self.indices[:,b] = ind_b
            #features_b = np.random.choice(np.arange(0,num_max_features,1),replace=False, size =num_max_features).astype(int)
            X_b = X[ind_b]
            y_b = y[ind_b]
            t = AbuTree(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity).fit(X_b,y_b)
            self.forest.append(t)
        return self
    
    def update(self, X: np.ndarray, y: np.ndarray):
        assert len(self.forest)>=1
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        
        n2 = n - self.indices.shape[0]
        indices = np.zeros((n2,self.n_estimators),dtype=int)
        for i,t in enumerate(self.forest):
            ind_b1 = self.indices[:,i]
            ind_b2 = np.random.choice(np.arange(n2,n,1,dtype=int),replace=True, size =n2 )
            indices[:,i] = ind_b2
            ind_b = np.concatenate((ind_b1,ind_b2), axis=0)
            X_b = X[ind_b]
            y_b = y[ind_b]
            self.forest[i].update(X_b,y_b)
        self.indices = np.vstack((self.indices,indices))
        return self
    


class ReevaluateRandomForest(RandomForest):
    def __init__(self,n_estimators:int,max_features:str = "all", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:

        super().__init__(n_estimators,max_features,criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        
    def fit(self,X,y):
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        self.forest = []
        for b in range(self.n_estimators):
            ind_b = np.random.choice(np.arange(0,n,1,dtype=int),replace=True, size =n )
            #features_b = np.random.choice(np.arange(0,num_max_features,1),replace=False, size =num_max_features).astype(int)
            X_b = X[ind_b]
            y_b = y[ind_b]
            t = TreeReevaluation(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity, random_state = self.random_state).fit(X_b,y_b)
            self.forest.append(t)
        return self
    
    def update(self, X: np.ndarray, y: np.ndarray):
        assert len(self.forest)>=1
        np.random.seed(self.random_state)
        n,num_features = X.shape
        num_max_features = max_features_to_int[self.max_features](num_features)
        for i,t in enumerate(self.forest):
            ind_b = np.random.choice(np.arange(0,n,1,dtype=int),replace=True, size =n )
            X_b = X[ind_b]
            y_b = y[ind_b]
            self.forest[i].update(X_b,y_b)
        return self
    






            
########

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import BaseEnsemble
import numpy as np
from joblib import Parallel, delayed
class MyRandomForestRegressor(RandomForestRegressor, RegressorMixin):
    def __init__(self, method="base", n_estimators=10, criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                  max_features:int = "third", random_state : int = None, gamma:float = 0.5,alpha:float = 0.0, delta:float = 0.05, max_samples=None):
        
        
        self.max_samples = max_samples
        super().__init__( n_estimators=n_estimators,random_state= random_state)
        self.estimator = MyDecisionTreeRegressor
        self.estimators_ = [self.estimator(method,criterion,max_depth= max_depth,min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf, adaptive_complexity=adaptive_complexity,
                                                  max_features= max_features,random_state=i,gamma = gamma,alpha=alpha,delta=delta) for i in range(n_estimators)]

    def fit_estimator(self, X, y, estimator):
            return estimator.fit(X, y)
    
    def fit(self, X, y):
        # Split the training data into random subsets for each estimator
        n_samples = X.shape[0]
        max_samples = self.max_samples if self.max_samples is not None else n_samples
        sample_indices = [np.random.choice(n_samples, max_samples, replace=True) for _ in range(self.n_estimators)]
        
        # self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        #     delayed(self.fit_estimator)(X[sample_indices[i]], y[sample_indices[i]], estimator)
        #     for i, estimator in enumerate(self.estimators_))

        # Fit each estimator to its random subset of the training data
        for i, estimator in enumerate(self.estimators_):
            estimator.fit(X[sample_indices[i]], y[sample_indices[i]])

    def update(self, X, y):
        # Split the training data into random subsets for each estimator
        n_samples = X.shape[0]
        max_samples = self.max_samples if self.max_samples is not None else n_samples
        sample_indices = [np.random.choice(n_samples, max_samples, replace=True) for _ in range(self.n_estimators)]
        #Parallel(n_jobs=-1)(delayed(estimator.fit)(X[sample_indices[i]], y[sample_indices[i]]) for i, estimator in enumerate(self.estimators_))

        # # Fit each estimator to its random subset of the training data
        for i, estimator in enumerate(self.estimators_):
            estimator.update(X[sample_indices[i]], y[sample_indices[i]])

    def predict(self, X):
        # Predict the target values for the test data using each estimator in the ensemble
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])

        # Return the average of the predictions
        return np.mean(predictions, axis=0)
    
class MyDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,method:str = "base", criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                  max_features:int = None, random_state : int = None, gamma:float = 0.5,alpha:float = 0.0, delta:float = 0.05):
        criterion = str(criterion).lower()
        if criterion not in criterions.keys():
            raise ValueError("Possible criterions are 'mse' and 'poisson'.")
        self.criterion = criterion
        self.method = method
        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)

        self.max_features = max_features

        self.min_samples_split = float(min_samples_split)

        if random_state is None:
            random_state = 0
        self.random_state = int(random_state)
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta

        self.adaptive_complexity = adaptive_complexity
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = 1
        self.tree = None

    def fit(self, X, y):
        max_feature = max_features_to_int[self.max_features](X.shape[1])
        if self.method  == "base":
           self.tree = BaseLineTree(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity,max_features=max_feature)
        elif self.method  == "nu":
            self.tree = NaiveUpdate(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity,max_features=max_feature)
        elif self.method  == "tr":
            self.tree = TreeReevaluation(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity,alpha=self.alpha, delta=self.delta,max_features=max_feature)
        elif self.method  == "sl":
            self.tree = StabilityRegularization(criterion = self.criterion,max_depth= self.max_depth,
                              min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                adaptive_complexity=self.adaptive_complexity, gamma = self.gamma,max_features=max_feature)
        elif self.method  == "abu":
            self.tree = AbuTree(criterion = self.criterion,max_depth= self.max_depth,
                                min_samples_split= self.min_samples_split,min_samples_leaf = self.min_samples_leaf,
                                    adaptive_complexity=self.adaptive_complexity,max_features=max_feature)
        elif self.method  == "babu":
            self.tree = BABUTree(criterion = self.criterion,
                                 max_depth= self.max_depth,
                                min_samples_split= self.min_samples_split,
                                min_samples_leaf = self.min_samples_leaf,
                                    adaptive_complexity=self.adaptive_complexity,bumping_iterations=3,max_features=max_feature)
        else:
            raise Exception("No tree found!")
        
        self.tree.fit(X, y)
        return self
    
    def update(self, X, y):
        if self.tree is None:
            raise Exception("No tree found!")
        self.tree.update(X, y)
        return self

    def predict(self, X):
        return self.tree.predict(X)

if __name__ == '__main__':
    # Define the training and test data
    n = 1000
    from sklearn.model_selection import train_test_split
    np.random.seed(0)
    X = np.random.uniform(size=(n,2),low=0,high=4)
    y = np.random.normal(loc=X[:,0]+X[:,1],scale=3,size = n)
    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)
    X_test = np.random.uniform(size=(n,2),low=0,high=4)
    y_test = np.random.normal(loc=X_test[:,0]+X_test[:,1],scale=3,size = n)


    import time

    start_time = time.time()
    rf = MyRandomForestRegressor(method="base", n_estimators=200, adaptive_complexity=True, random_state=0).fit(X1, y1)
    end_time = time.time()
    elapsed_time = end_time - start_time
   

    print("Elapsed time: ", elapsed_time, "seconds")
    
    start_time = time.time()
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X1, y1)
    end_time = time.time()
    elapsed_time = end_time - start_time

    

    print("Elapsed time: ", elapsed_time, "seconds")


    start_time = time.time()
    rf = RF("base",n_estimators= 200,max_features="third",criterion="mse",adaptive_complexity=True)
    end_time = time.time()
    elapsed_time = end_time - start_time

    

    print("Elapsed time: ", elapsed_time, "seconds")
    # # Create a Random Forest Regressor with our custom base estimator
    # rf = MyRandomForestRegressor(method="base", n_estimators=200, adaptive_complexity=True, random_state=0)


    # # Fit the model to the training data
    # rf.fit(X1, y1)
    # rf.update(X, y)

    # # Predict the target values for the test data
    # y_pred = rf.predict(X)

    # # Print the predicted values
    # print(y_pred)
    # from matplotlib import pyplot as plt
    # plt.subplot(1,3,1)
    # plt.scatter(X_test[:,0],y_test)

    # plt.scatter(X[:,0],y_pred[:],c ="red")
    # plt.title("my impl")

    # plt.subplot(1,3,2)
    # plt.scatter(X_test[:,0],y_test)

    # rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X1, y1)
    # y_pred = rf.predict(X_test)
    # plt.scatter(X_test[:,0],y_pred[:],c ="red")
    # plt.title("sklearn")
    # plt.subplot(1,3,3)
    # rf = RandomForestRegressor(n_estimators=200,min_samples_leaf=5, random_state=0).fit(X1, y1)
    # y_pred = rf.predict(X_test)
    # plt.scatter(X_test[:,0],y_test)
    # plt.scatter(X_test[:,0],y_pred[:],c ="red")
    # plt.title("sklearn")

    # plt.show()

    # rf = MyRandomForestRegressor(method="base", n_estimators=200, adaptive_complexity=True, random_state=0)


    # # Fit the model to the training data
    # rf.fit(X1, y1)
    # rf.update(X, y)

    # # Predict the target values for the test data
    # y_pred = rf.predict(X_test)

    # # Print the predicted values
    # print(y_pred)
    # from matplotlib import pyplot as plt
    # plt.subplot(1,3,1)
    # errors = (y_test - y_pred)**2
    # mean_error = np.mean(errors)
    # plt.hist(errors)
    # plt.axvline(mean_error, color='r', linestyle='--')
    # plt.text(mean_error+0.1, 60, f'Mean = {mean_error:.2f}')
    # plt.title("sklearn")
    # plt.title("my impl")

    # plt.subplot(1,3,2)


    # rf = MyRandomForestRegressor(method="babu", n_estimators=200, adaptive_complexity=True, random_state=0)
    # rf.fit(X1, y1)
    # rf.update(X, y)
    # y_pred = rf.predict(X_test)
    # errors = (y_test - y_pred)**2
    # mean_error = np.mean(errors)
    # plt.hist(errors)
    # plt.axvline(mean_error, color='r', linestyle='--')
    # plt.text(mean_error+0.1, 60, f'Mean = {mean_error:.2f}')
    # plt.title("nu")
    # plt.subplot(1,3,3)
    # rf = RandomForestRegressor(n_estimators=200,min_samples_leaf=5, random_state=0).fit(X, y)
    # y_pred = rf.predict(X_test)
    # errors = (y_test - y_pred)**2
    # mean_error = np.mean(errors)
    # plt.hist(errors)
    # plt.axvline(mean_error, color='r', linestyle='--')
    # plt.text(mean_error+0.1, 60, f'Mean = {mean_error:.2f}')
    # plt.title("sklearn")


    # plt.show()




        


