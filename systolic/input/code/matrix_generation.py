# MPI_systolic_arrays/input/code/generate_matrices.py
import numpy as np
from pathlib import Path

# directory dello script: .../MPI_systolic_arrays/input/code
SCRIPT_DIR = Path(__file__).resolve().parent
# vogliamo salvare in .../MPI_systolic_arrays/input/matrices
OUTDIR = SCRIPT_DIR.parent / "matrices"
OUTDIR.mkdir(parents=True, exist_ok=True)

def save_with_trailing_commas(path, M, fmt="%.6f"):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in M:
            f.write(",".join(fmt % v for v in row) + ",\n")

def gen_and_save(size, low=0.0, high=100.0):
    A = np.random.uniform(low, high, size=(size, size)).astype(np.float64)
    B = np.random.uniform(low, high, size=(size, size)).astype(np.float64)
    save_with_trailing_commas(OUTDIR / f"matrix_A_{size}.csv", A)
    save_with_trailing_commas(OUTDIR / f"matrix_B_{size}.csv", B)

if __name__ == "__main__":
    for n in (500, 1000, 2000):
        gen_and_save(n)
    print(f"OK: file creati in {OUTDIR}")

