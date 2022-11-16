
from cProfile import label


if __name__ == "__main__":
    from CONFIG import SEED, EPSILON
    from sklearn.datasets import make_regression,load_diabetes
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split,KFold
    from datasimulator import simulate_claim_frequencies
    import matplotlib.pyplot as plt
    
    import numpy as np 
    import sys
    import os
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cur_file_path + '\\..\\..\\cpp\\build\\Release\\')
    sys.path.append(cur_file_path + '\\..')

    from tree.RegressionTree import BaseLineTree, StableTree1,sklearnBase,StableTree2,StableTree3, StableTree4
    import pandas as pd
    N = 2000
    dataset_name =  "make_regression"  #"sim_claim_freq" #"diabetes" #
    if dataset_name == "sim_claim_freq":
        data = simulate_claim_frequencies(N)
        data = pd.DataFrame(data.groupby(['obj_id',"obj_age", "obj_size",	"obj_type_1", "obj_type_2", "obj_type_3"]).sev.count()-1).reset_index().rename(columns={'sev': 'freq'}).drop("obj_id", axis=1).to_numpy()
        X,y = data[:,:-1], data[:,-1]
    elif dataset_name == "make_regression":
        X,y= make_regression(N,10, random_state=SEED)
        y = y + np.max(y)+100
    elif dataset_name == "diabetes":
        X,y = load_diabetes(return_X_y=True)
        N = len(y)


    min_samples_split = 4
    kf = KFold(n_splits=X.shape[0], random_state=SEED, shuffle=True)
    
    models = {  
                 "baseline": BaseLineTree(min_samples_split=min_samples_split),
                 "sklearn": sklearnBase(min_samples_split=min_samples_split, random_state=0),
                 "method1":StableTree1(delta=0.0001, min_samples_split=min_samples_split)
                 #"method2":StableTree2(min_samples_split=min_samples_split),
                 #"method3":StableTree3(min_samples_split=min_samples_split)
                #"#method4":StableTree4(min_samples_split=min_samples_split)
            }
    stability = {name:[] for name in models.keys()}
    standard_stability = {name:[] for name in models.keys()}
    mse = {name:[] for name in models.keys()}
    iteration = 1
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
            # if iteration==482:
            #     print(np.apply_along_axis(lambda x:len(np.unique(x)) , 0, X1) )
            #     print(np.unique(y1, return_counts=True))
            #     print(np.any(np.isnan(y1)))
            #     print(np.any(np.isinf(y1)))
            #     print(np.apply_along_axis(lambda x:np.any(np.isnan(x)), 0, X1) )
            #     print(np.apply_along_axis(lambda x:np.any(np.isinf(x)), 0, X1) )
                
            #print("before")
            model.fit(X1,y1)
            #print("after")
        
            pred1 = model.predict(X_test)
            model.update(X_12,y_12)
            pred2 = model.predict(X_test)
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(np.log((pred1.item()+EPSILON)/(pred2.item()+EPSILON)))
            standard_stability[name].append(abs(pred1.item()- pred2.item()))
        
            if (iteration) % 100 ==0:
                print(f"{iteration}/{N}, {name} - mse: {np.mean(mse[name]):.3f}, stability: {np.std(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")

        iteration+=1

    print(models)
    for name in models.keys():
        print("="*80)
        print(f"{name} - mse: {np.mean(mse[name]):.3f}, stability: {np.std(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")
        print("="*80)   

    plt.rcParams["figure.figsize"] = (16,8)
    labels, data = stability.keys(), stability.values()
    
    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel(r'$\log\left(\frac{T_1(x_i)}{T_2(x_i)}\right)$',fontsize=12)
    plt.xlabel("models")
    plt.savefig(f"python\examples\\plots\\{dataset_name}_stability.png")
    plt.close()


    labels, data = standard_stability.keys(), standard_stability.values()
    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel(r'$| T_1(x_i)-T_2(x_i) |$',fontsize=12)
    plt.xlabel("models")
    plt.savefig(f"python\examples\\plots\\{dataset_name}_standard_stability.png")
    plt.close()

    
    labels, data = mse.keys(), mse.values()
    plt.boxplot(data)
    plt.ylabel('mse',fontsize=12)
    plt.xlabel("models")
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.savefig(f"python\examples\\plots\\{dataset_name}_mse.png")
    plt.close()
