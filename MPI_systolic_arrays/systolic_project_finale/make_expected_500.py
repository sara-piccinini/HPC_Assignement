import numpy as np

N = 500

A = np.loadtxt("input_matrices/A_{}.csv".format(N), delimiter=",")
B = np.loadtxt("input_matrices/B_{}.csv".format(N), delimiter=",")

C = A @ B

np.savetxt("C_expected_{}.csv".format(N), C, delimiter=",")

