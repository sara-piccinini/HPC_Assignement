import numpy as np

matrix_1 = np.random.uniform(0.0, 100.0, size=(2000, 2000)).astype(np.float64)
matrix_2 = np.random.uniform(0.0, 100.0, size=(2000, 2000)).astype(np.float64)

np.savetxt("matrix_A_2000.csv", matrix_1, delimiter =",", fmt="%.3f")
np.savetxt("matrix_B_2000.csv", matrix_2, delimiter =",", fmt="%.3f")
# This script generates two random matrices of size 2000x2000 with values between 0.0 and 100.0,
# saves them to CSV files with three decimal places, and uses float64 data type.
# The matrices are saved as 'matrix_A_2000.csv' and 'matrix_B_2000.csv'.