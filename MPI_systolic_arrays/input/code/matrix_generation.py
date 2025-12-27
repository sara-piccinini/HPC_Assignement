import numpy as np

# Generates two matrices with float64 (double precision) data
matrix1 = np.random.uniform(0.0, 100.0, size=(2000, 2000)).astype(np.float64)
matrix2 = np.random.uniform(0.0, 100.0, size=(2000, 2000)).astype(np.float64)

# Saves them in file .csv with 3 decimal digits
np.savetxt("matrice_A_2000.csv", matrix1, delimiter=",", fmt="%.3f")
np.savetxt("matrice_B_2000.csv", matrix2, delimiter=",", fmt="%.3f")
