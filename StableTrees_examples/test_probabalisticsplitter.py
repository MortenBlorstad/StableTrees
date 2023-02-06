from _stabletrees import ProbabalisticSplitter,Splitter
from _stabletrees import ProbabalisticTree as probtree
from stabletrees import ProbabalisticTree, BaseLineTree,EvoTree

from sklearn.datasets import make_regression, load_diabetes
from matplotlib import pyplot as plt
X,y = make_regression(100,n_features=5,n_informative=3,noise=10,random_state=0)

X,y = load_diabetes(return_X_y=True)
print(X.shape)

from sklearn.model_selection import train_test_split

# for i in range(100):
#     x1,x2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)
#     t = BaseLineTree().fit(x1,y1)
    


ps = ProbabalisticSplitter(1,1000,0, False,0 )
s = Splitter(1,1000,0, False )
import numpy as np

#print(np.corrcoef(np.concatenate([X,y.reshape(-1,1)], axis=1), rowvar=False))

# res = []
# res2 = []
# for i in range(10000):
#     _, f,_,_,_ = ps.find_best_split(X,y)
#     tree = ProbabalisticTree(0,1,2,1,False, i*31 + i%13 )
#     tree.learn(X,y)
#     root = tree.get_root()
#     res2.append(root.get_split_feature())
#     tree2 = ProbabalisticTree(0,1,2,1,False, i )
#     tree2.learn(X,y)
#     root2 = tree2.get_root()
#     res2.append(root.get_split_feature())
#     res.append(root2.get_split_feature())





# tree.learn(X,y)

# tree.update(X,y)






# print(copied.get_split_feature())
# node = copied.get_left_node()
# print(node.get_split_feature())
# node = node.get_left_node()
# print(node.predict())




# tree.plot()
# plt.show()




# t = ProbabalisticTree(max_depth=2,min_samples_split= 2, min_samples_leaf= 1).fit(X,y)
# print(t.tree.make_node_list())
# t.plot()
# plt.show()

# tree1 = probtree(0,3,2,1,False,0)
# tree1.learn(X,y)
# tree2 = probtree(0,3,2,1,False,1)
# tree2.learn(X,y)

#print(["None" if n is None else n.toString() for n in tree2.make_node_list()] )
# print(tree2.get_root().get_left_node().toString())
# print(tree2.get_root().copy().get_left_node().toString())

#tree2 = tree2.copy()
# print()
# print(["None" if n is None else n.toString() for n in tree2.make_node_list()] )
x1,x2,y1,y2 = train_test_split(X,y,test_size=0.15,random_state=0)

tree = EvoTree(max_depth = 5, random_state = 0)
tree.fit(x1,y1)
pred = tree.predict(x2)
print(np.mean((y2 - pred)**2))
tree.plot()
plt.show()
# print("sd")
# pop = tree.create_population(X,y,10)

#tree.fitness_function(pop,X,y,tree.predict(X))

# print(len(pop))
# for p in pop:
#     print(np.mean((y - p.predict(X))**2))

# print("sdad")
# tree.fitness_function(pop,X,y,tree.predict(X))
# for p in pop:
#     print(np.mean((y - p.predict(X))**2))

# pop = tree.generate_population(X,y,pop,10,0.5, 0.25)
# type(pop)


# print("sda")
# print(len(pop))
# for p in pop:
#     print(np.mean((y - p.predict(X))**2))

# print("start")
# tree.update(x1,y1,400,500)
# print("done")
# newpred= tree.predict(x2)
# print(np.mean((y2 - pred)**2))
# print(np.mean((y2 - newpred)**2))
# print(np.std((pred+1.1)/(newpred+1.1)))
# print(np.mean(abs(pred - newpred)))

# t1,t2 = tree.breed(X,y,tree1,tree2)

# t1,t2 = tree.breed(X,y,t1,t2)


# t1,t2 = tree.breed(X,y,tree1,tree1)


tree.plot()
plt.show()

# t1,t2 = tree.breed(X,y,t1,tree2)

# print("sdad")

# tree.root = t1.get_root()



# tree.plot()
# plt.show()




# print("crossover start")
# tree1.crossover(X,y,tree2.root.get_left_node().copy() ,1)

# print("sadasd")

# tree2.plot()
# plt.show()


# plt.subplot(1,2,1)
# plt.hist(res)
# plt.subplot(1,2,2)
# plt.hist(res2)
# plt.show()