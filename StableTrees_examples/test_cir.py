from stabletrees import rnchisq,cir_sim_vec,cir_sim_mat
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)
nsims = 3
mat = cir_sim_mat(nsims,100)

for i in range(nsims):
    plt.plot(mat[i,:])
plt.show()

mat = cir_sim_mat(nsims,100)

for i in range(nsims):
    plt.plot(mat[i,:])
plt.show()
mat = cir_sim_mat(nsims,100)
for i in range(nsims):
    plt.plot(mat[i,:])
plt.show()