import numpy as np  
from stabletrees import BaseLineTree,AbuTree,AbuTreeI
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
if __name__ == "__main__":
    np.random.seed(0)
    # X  = np.array([[1.1,1],[1,2],[2.1,4],[2,3],[4,5],[6,10], [4,6]])
    # y = X[:,0]*2 + X[:,1]*0.5 + np.random.normal(0,0.5, X.shape[0])
    from sklearn.datasets import load_boston,load_diabetes
    from sklearn.model_selection import train_test_split,RepeatedKFold
    


    data = pd.read_csv("data/"+ "boston" +".csv") # load dataset
    target = "medv"
    feature =  [ "crim", "rm"]
    data = data.dropna(axis=0, how="any") # remove missing values if any
    data = data.loc[:, feature + [target]] # only selected feature and target variable
    y = data[target].to_numpy()
    X = data.drop(target, axis=1).to_numpy()

    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=0)
    model = AbuTreeI()
    i =0
    for train_index, test_index in kf.split(X):        
        if i ==12:
            X_12, y_12 = X[train_index],y[train_index]
            X_test,y_test = X[test_index],y[test_index]

        i +=1
        
    X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=0)   
    model.fit(X1,y1)  
    pred1 = model.predict(X_test)
    pred1_train = model.predict(X_12)
    pred1_orig= model.predict(X1)
    print(X_12.shape)
    print("before")
    print("\n"*10)
    model.update(X_12,y_12)

    print("after")
    model.plot()
    plt.show()
    #X,y = load_diabetes(return_X_y=True)
    
    
    # df = pd.read_csv("C:\\Users\\mb-92\\OneDrive\\Skrivebord\\studie\\StableTrees\\StableTrees_examples\\test_data.csv")
    # X = df["x"].to_numpy().reshape(-1,1)
    # y = df["y"].to_numpy()
    # # n = 1000
    # # X =np.random.uniform(size=(n,1), low = 0,high = 4)
    # # y =np.random.normal(X.ravel(),1, size=n)
    # print(2*(y.mean() - y).sum())
    # #for i in range(100):
    # X1,X2,y1,y2 = train_test_split(X,y,test_size=0.25,random_state=0)
    # #X2 = X[0:250,:]; y2 = y[0:250]
    # print(X2.shape)
    #tree = AbuTreeI().fit(X1,y1)
    #tree = AbuTree().fit(X,y)
    #print(i,"update")
    # treeI.plot()
    # plt.show()
   

    # tree.plot()
    # plt.show()


    # tree = AbuTreeI().fit(X1,y1)
    
    # ypred = tree.predict(X1)
    # plt.subplot(1,2,1)
    # plt.scatter(X1,y1)
    # plt.scatter(X1,ypred,c ="red")
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.title("Prior tree")
    # tree = tree.update(X2,y2)
    # ypred = tree.predict(X2)
    
    # plt.subplot(1,2,2)
    # plt.scatter(X2,y2)
    
    # plt.scatter(X2,ypred,c ="red")
    
    # plt.xlabel(f"X2")
    # plt.ylabel(f"y2")
    # plt.title(f"Posterior tree")
    # ypred = tree.predict(X)
    # print(mean_squared_error(y,ypred))
    
    # plt.show()

    # tree = AbuTreeI().fit(X1,y1)
    # ypred = tree.predict(X)
    # plt.subplot(1,4,1)
    # plt.scatter(X,y)
    # plt.scatter(X,ypred,c ="red")
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.title("Prior tree")

    # ypred = tree.predict(X)
    # print(mean_squared_error(y,ypred))
    
    
  
    # t = AbuTreeI().fit(X2,y2)
    # ypred = t.predict(X2)
    # plt.subplot(1,4,2)
    # plt.scatter(X2,y2)
    # plt.scatter(X2,ypred,c ="red")
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.title("Train 2 - only D2")
    # ypred = t.predict(X)
    # print(mean_squared_error(y,ypred))

    
    # tree = tree.update(X2,y2)
    # ypred = tree.predict(X2)
    

    # ypred = tree.predict(X2)
    
    # plt.subplot(1,4,3)
    # plt.scatter(X2,y2)
    # plt.scatter(X2,ypred,c ="red")
    
    # plt.xlabel(f"X2")
    # plt.ylabel(f"y2")
    # plt.title(f"Posterior tree")
    # ypred = tree.predict(X)
    # print(mean_squared_error(y,ypred))
    
    

    # t = AbuTreeI().fit(X,y)
    # ypred = t.predict(X)
    # print(mean_squared_error(y,ypred))
    # plt.subplot(1,4,4)
    # plt.scatter(X,y)
    # plt.scatter(X,ypred,c ="red")
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.title("Train 2 - D1 & D2")

    # plt.show()

# ##
#     tree = AbuTree().fit(X,y)
#     ypred = tree.predict(X)
#     print(1,mean_squared_error(y,ypred))
#     plt.subplot(2,4,1)
#     plt.scatter(X,y)
#     plt.scatter(X,ypred,c ="red")
#     plt.xlabel("X")
#     plt.ylabel("y")
#     plt.title("Prior tree")

#     t = BaseLineTree(adaptive_complexity=True).fit(X,y)
#     ypred = t.predict(X)
#     print(5,mean_squared_error(y,ypred))
#     plt.subplot(2,4,5)
#     plt.scatter(X,y)
#     plt.scatter(X,ypred,c ="red")
#     plt.xlabel("X")
#     plt.ylabel("y")
#     plt.title("standard tree")
   
#     n = 500
#     np.random.seed(123)
#     for i,g in enumerate(range(3)):
#         X =np.random.uniform(size=(n,1), low = 0,high = 4)
#         y =np.random.normal(X.ravel(),1, size=n)
#         tree = tree.update(X,y)
#         ypred = tree.predict(X)
#         print(i+2,mean_squared_error(y,ypred))
#         plt.subplot(2,4,i+2)
#         plt.scatter(X,y)
#         plt.scatter(X,ypred,c ="red")
#         plt.xlabel(f"X{i+2}")
#         plt.ylabel(f"y{i+2}")
#         plt.title(f"Posterior tree {i+1}")
#         t = t.update(X,y)
#         ypred = t.predict(X)
#         print(4+i+2,mean_squared_error(y,ypred))
#         plt.subplot(2,4,4+i+2)
#         plt.scatter(X,y)
#         plt.scatter(X,ypred,c ="red")
#         plt.xlabel("X")
#         plt.ylabel("y")
#         plt.title("standard tree")
   
#     plt.show()
    
    # tree.plot()
    # plt.show()
    
    
        