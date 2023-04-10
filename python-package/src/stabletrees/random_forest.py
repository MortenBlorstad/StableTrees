
from stabletrees.tree import BaseRegressionTree,BaseLineTree,NaiveUpdate,AbuTreeI,TreeReevaluation
from _stabletrees import RandomForest as rf
import numpy as np

max_features_to_int = {"sqrt": np.sqrt,
                       "log2": np.log2,
                       None : len}

method_to_int = {"base":0,
                 "nu":1,
                 "tr":2,
                 "sr":3,
                 "abu":4}
criterions = {"mse":0, "poisson":1}




class RandomForest(BaseRegressionTree):
    def __init__(self,n_estimators:int,max_features:str = "sqrt", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        assert n_estimators>=1
        self.n_estimators = n_estimators
        self.max_features = max_features
        if self.max_features not in max_features_to_int.keys():
            self.max_features = "sqrt"
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
    

class RF(BaseRegressionTree):
    def __init__(self,method:str = "base",n_estimators:int = 100,max_features:str = "sqrt", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        assert n_estimators>=1
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.method = method
        self.criterion =criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        if self.max_features not in max_features_to_int.keys():
            self.max_features = "sqrt"

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)
        self.forest = None
        

    def fit(self,X,y):
        self.forest = rf(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,False,X.shape[1]//3,method_to_int[self.method])
        self.forest.learn(X,y)
        return self
        
    def update(self, X: np.ndarray, y: np.ndarray):
        if self.forest is None:
            return self.fit(X,y)
        self.forest.update(X,y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert (self.forest is not None)
        return self.forest.predict(X)
        


class NaiveRandomForest(RandomForest):
    def __init__(self,n_estimators:int,max_features:str = "sqrt", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:

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
    def __init__(self,n_estimators:int,max_features:str = "sqrt", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:

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
            t = AbuTreeI(criterion = self.criterion,max_depth= self.max_depth,
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
    def __init__(self,n_estimators:int,max_features:str = "sqrt", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:

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
    






            




        


