import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: python3 check_results.py <test_output.csv> <expected_output.csv>")
    sys.exit(1)

C_test = np.loadtxt(sys.argv[1], delimiter=",")
C_ref = np.loadtxt(sys.argv[2], delimiter=",")

error = np.max(np.abs(C_test - C_ref))
print("Max absolute error: {}".format(error))

