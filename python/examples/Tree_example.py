

if __name__ == "__main__":
    from CONFIG import SEED, EPSILON
    from sklearn.datasets import make_regression,load_diabetes
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split,RepeatedKFold
    from datasimulator import simulate_claim_frequencies
    import matplotlib.pyplot as plt
    
    import numpy as np 
    import sys
    import os
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cur_file_path + '\\..\\..\\cpp\\build\\Release\\')
    sys.path.append(cur_file_path + '\\..')

    from tree.RegressionTree import BaseLineTree, StableTree1,sklearnBase,StableTree2,StableTree3, StableTree4,StableTreeM2
    import pandas as pd

    N = 2000
    dataset_name =  "sim_claim_freq" #"diabetes" #"make_regression"  #
    if dataset_name == "sim_claim_freq":
        data = simulate_claim_frequencies(N)
        data = pd.DataFrame(data.groupby(['obj_id',"obj_age", "obj_size",	"obj_type_1", "obj_type_2", "obj_type_3"]).sev.count()-1).reset_index().rename(columns={'sev': 'freq'}).drop("obj_id", axis=1).to_numpy()
        X,y = data[:,:-1], data[:,-1]
    elif dataset_name == "make_regression":
        X,y= make_regression(N,10, random_state=SEED, noise=10)
        y = y + np.max(np.abs(y))
    elif dataset_name == "diabetes":
        X,y = load_diabetes(return_X_y=True)
        N = len(y)

    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeRegressor
    parameters = {'max_depth':[None, 5, 10], 'min_samples_split':[2,4,8]}
    clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)
    clf.fit(X,y)
    params = clf.best_params_
    print(params)
    models = {  
                 "baseline": BaseLineTree(),
                 "sklearn": sklearnBase(random_state=0),
                 "method2":StableTreeM2(),
                 #"method1":StableTree1(delta=0.0001, min_samples_split=min_samples_split)
                 #"method2":StableTree2(min_samples_split=min_samples_split),
                 #"method3":StableTree3(min_samples_split=min_samples_split)
                #"#method4":StableTree4(min_samples_split=min_samples_split)
            }
    stability = {name:[] for name in models.keys()}
    standard_stability = {name:[] for name in models.keys()}
    mse = {name:[] for name in models.keys()}
    train_stability = {name:[] for name in models.keys()}
    train_standard_stability = {name:[] for name in models.keys()}
    train_mse = {name:[] for name in models.keys()}
    iteration = 1
    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
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
            pred1_train = model.predict(X_12)
            model.update(X_12,y_12)
            pred2 = model.predict(X_test)
            pred2_train =  model.predict(X_12)
            train_mse[name].append(mean_squared_error(pred2_train,y_12))
            train_stability[name].append(np.std(np.log((pred1_train+EPSILON)/(pred2_train+EPSILON))))
            train_standard_stability[name].append(np.mean(abs(pred1_train- pred2_train)))
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(np.std(np.log((pred1+EPSILON)/(pred2+EPSILON))))
            standard_stability[name].append(np.mean(abs(pred1- pred2)))
        
            if (iteration) % 10 ==0:
                print(f"{iteration}/{50}, {name}")
                print(f"train - mse: {np.mean(train_mse[name]):.3f}, stability: {np.mean(train_stability[name]):.3f}, standard stability: {np.mean(train_standard_stability[name]):.3f}")
                print(f"test - mse: {np.mean(mse[name]):.3f}, stability: {np.mean(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")
        iteration+=1

    print(models)
    for name in models.keys():
        print("="*80)
        print(f"{name}")
        print(f"train - mse: {np.mean(train_mse[name]):.3f}, stability: {np.mean(train_stability[name]):.3f}, standard stability: {np.mean(train_standard_stability[name]):.3f}")
        print(f"test - mse: {np.mean(mse[name]):.3f}, stability: {np.mean(stability[name]):.3f}, standard stability: {np.mean(standard_stability[name]):.3f}")
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
