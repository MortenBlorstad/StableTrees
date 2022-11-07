
if __name__ == "__main__":
    from CONFIG import SEED, EPSILON
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split,KFold
    import numpy as np 
    import sys
    import os
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cur_file_path + '\\..\\..\\cpp\\build\\Release\\')
    sys.path.append(cur_file_path + '\\..')

    from tree.RegressionTree import BaseLineTree, StableTree1,sklearnBase
    X,y= make_regression(2000,10, random_state=SEED)
    y = y + np.max(y)+100

    kf = KFold(n_splits=X.shape[0], random_state=SEED, shuffle=True)
    stability = {"method1": [],"method2": [], "baseline":[]}
    mse = {"method1": [],"method2": [], "baseline":[]}
    models = {"method1":StableTree1(delta=0.0001, min_samples_split=2), "baseline": BaseLineTree(min_samples_split=2),
            "method2": sklearnBase(min_samples_split=2, random_state=0)}
    iteration = 0
    # X1,X2,y1,y2 =  train_test_split(X, y, test_size=0.5, random_state=SEED)
    # model = BaseTree().fit(X1,y1)
    # y_pred = model.predict(X2)
    #
    #print(mean_squared_error(y2,y_pred))

    for train_index, test_index in kf.split(X):
        X_12, y_12 = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)

        
        # initial model 
        
        for name, model in models.items():
            model.fit(X1,y1)
            pred1 = model.predict(X_test)
            model.update(X_12,y_12)
            pred2 = model.predict(X_test)
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(np.log((pred1.item()+EPSILON)/(pred2.item()+EPSILON)))
       
            if iteration % 100 ==0:
               print(f"{iteration}/2000, {name} - stability: {np.std(stability[name]):.3f}, mse: {np.mean(mse[name]):.3f}")
        
        iteration+=1


    for name in models.keys():
        print("="*40)
        print(f"{name} - stability: {np.std(stability[name]):.3f}, mse: {np.mean(mse[name]):.3f}")
        print("="*40)   



