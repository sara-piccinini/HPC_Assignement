import numpy as np

def save_matrix(filename, mat):
    np.savetxt(filename, mat, delimiter=",", fmt="%.4f")

size = 2000  
#size is the parameter which indicates the size of A and B matrices
A = np.random.rand(size, size)
B = np.random.rand(size, size)

save_matrix("A_{}.csv".format(size), A)
save_matrix("B_{}.csv".format(size), B)

