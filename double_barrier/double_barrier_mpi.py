# double_barrier_mpi.py
# Transmission through a double barrier (barrier–well–barrier) using MPI (mpi4py)
# Example setup: V=0.1 eV, L1=L3=100 Å, L2 in {20,50,100} Å, m*=0.067 m_e (GaAs)
# We compute T(E) in two equivalent ways:
#   - via Transfer matrix (T-matrix):   T = 1 / |T22|^2
#   - via Scattering matrix (S-matrix): T = |S21|^2
# The two columns must agree (up to small numerical error) because they describe the same physics.

from mpi4py import MPI
import numpy as np
import argparse

# --- Physical constants (SI) ---
HBAR = 1.054571817e-34    # J·s
QE   = 1.602176634e-19    # J/eV
M_E  = 9.1093837015e-31   # kg
ANG  = 1e-10              # m

# ---------- Basic 1D piecewise-constant model ----------
def k_from(E_eV, V_eV, m_eff):
    """
    Complex wave number k [1/m] in a region with constant potential V
    and effective mass m_eff (in units of the electron mass).
    k = sqrt(2 m* (E - V)) / hbar ; becomes imaginary if E < V.
    """
    m  = m_eff * M_E
    dE = (E_eV - V_eV) * QE
    return np.sqrt(2.0 * m * (dE + 0j)) / HBAR

def interface_matrix_T(kL, kR):
    """
    Interface (left -> right) for plane waves with the *same* effective mass on both sides.
    Amplitude basis: [A_fwd, A_bwd]. Convention:
        [A_R; B_R] = M_if * [A_L; B_L]
    """
    if abs(kL) == 0:
        kL = 1e-30  # avoid division by zero
    r = kR / kL
    return 0.5 * np.array([[1 + r, 1 - r],
                           [1 - r, 1 + r]], dtype=complex)

def propagation_matrix_T(k, a_m):
    """
    Propagation inside a homogeneous layer of thickness a:
        diag(e^{i k a}, e^{-i k a})
    """
    phase = np.exp(1j * k * a_m)
    return np.array([[phase, 0.0],
                     [0.0,   1.0/phase]], dtype=complex)

def S_from_T(M):
    """
    Convert total Transfer matrix M = [[A,B],[C,D]] to S-matrix (identical leads).
        r  = C/D  = S11  (reflection from left)
        t  = 1/D  = S21  (transmission from left)  <-- used for T
        r' = -B/D = S22
        t' = 1/D  = S12
    """
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    if D == 0:
        return np.nan, np.nan, np.nan, np.nan
    r  = C / D
    t  = 1.0 / D
    rp = -B / D
    tp = 1.0 / D
    return r, t, rp, tp

def build_double_barrier_T(E_eV, V_eV, L1_A, L2_A, L3_A, m_eff):
    """
    Total T-matrix for: lead(L,V=0) -> barrier(L1,V=V_eV) -> well(L2,V=0)
                        -> barrier(L3,V=V_eV) -> lead(R,V=0)
    """
    kL = k_from(E_eV, 0.0,  m_eff)
    k1 = k_from(E_eV, V_eV, m_eff)  # barrier
    k2 = k_from(E_eV, 0.0,  m_eff)  # well
    kR = k_from(E_eV, 0.0,  m_eff)

    M = np.eye(2, dtype=complex)
    # L -> barrier 1
    M = interface_matrix_T(kL, k1) @ M
    M = propagation_matrix_T(k1, L1_A * ANG) @ M
    # barrier 1 -> well
    M = interface_matrix_T(k1, k2) @ M
    M = propagation_matrix_T(k2, L2_A * ANG) @ M
    # well -> barrier 3
    M = interface_matrix_T(k2, k1) @ M
    M = propagation_matrix_T(k1, L3_A * ANG) @ M
    # barrier 3 -> R
    M = interface_matrix_T(k1, kR) @ M
    return M

# ---------- Energy sweep (each MPI rank handles a slice of energies) ----------
def compute_sweep(Egrid, V_eV, L1_A, L2_A, L3_A, m_eff):
    """
    Returns two arrays with the same length as Egrid:
        T_T22  = 1/|T22|^2  (transfer-matrix definition)
        T_S21  = |S21|^2    (scattering-matrix definition)
    With identical leads the two definitions are equivalent.
    """
    T_T22 = np.zeros_like(Egrid, dtype=float)
    T_S21 = np.zeros_like(Egrid, dtype=float)

    for i, E in enumerate(Egrid):
        # Build total transfer matrix at this energy
        Mtot = build_double_barrier_T(E, V_eV, L1_A, L2_A, L3_A, m_eff)

        # 1) via T-matrix: D = T22
        D = Mtot[1,1]
        if abs(D) < 1e-300:
            T_T22[i] = 0.0
        else:
            T_T22[i] = 1.0 / (abs(D)**2)

        # 2) via S-matrix: use S21 = t
        r, t, rp, tp = S_from_T(Mtot)
        T_S21[i] = 0.0 if (isinstance(t, float) and np.isnan(t)) else float(abs(t)**2)

        # Optional: light clipping could be enabled after debugging
        # T_T22[i] = min(max(T_T22[i], 0.0), 1.0)
        # T_S21[i] = min(max(T_S21[i], 0.0), 1.0)

    return T_T22, T_S21

# -------------------- MAIN: MPI parallelization over energy --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emin",   type=float, default=0.0,   help="Min energy (eV)")
    ap.add_argument("--emax",   type=float, default=0.2,   help="Max energy (eV)")
    ap.add_argument("--nE",     type=int,   default=4000,  help="Number of energy samples")
    ap.add_argument("--V",      type=float, default=0.1,   help="Barrier height (eV)")
    ap.add_argument("--L1",     type=float, default=100.0, help="Barrier-1 thickness (Å)")
    ap.add_argument("--L3",     type=float, default=100.0, help="Barrier-3 thickness (Å)")
    ap.add_argument("--L2s",    type=str,   default="20,50,100", help="Comma-separated list of L2 (Å)")
    ap.add_argument("--meff",   type=float, default=0.067,  help="Effective mass (in units of m_e)")
    ap.add_argument("--prefix", type=str,   default="T_vs_E",     help="Output filename prefix")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Build the global energy grid on rank 0 and split among ranks
    if rank == 0:
        E = np.linspace(args.emin, args.emax, args.nE, dtype=float)
        chunks = np.array_split(E, size)
    else:
        chunks = None

    E_local = comm.scatter(chunks, root=0)

    L2_list = [float(x) for x in args.L2s.split(",") if x.strip()]
    for L2 in L2_list:
        # Local computation on each rank
        T_T22_local, T_S21_local = compute_sweep(E_local, args.V, args.L1, L2, args.L3, args.meff)

        # Gather results on rank 0
        gathered_E    = comm.gather(E_local,       root=0)
        gathered_TT22 = comm.gather(T_T22_local,   root=0)
        gathered_TS21 = comm.gather(T_S21_local,   root=0)

        if rank == 0:
            E_all    = np.concatenate(gathered_E)
            TT22_all = np.concatenate(gathered_TT22)
            TS21_all = np.concatenate(gathered_TS21)

            # Sort by energy (scatter may scramble order)
            idx = np.argsort(E_all)
            E_all, TT22_all, TS21_all = E_all[idx], TT22_all[idx], TS21_all[idx]

            # Save CSV: E_eV, T_from_T22, T_from_S21
            out = f"{args.prefix}_L2={int(L2)}A.csv"
            np.savetxt(out, np.column_stack([E_all, TT22_all, TS21_all]),
                       delimiter=",", header="E_eV,T_from_T22,T_from_S21", comments="")
            print(f"[L2={L2:>5.1f} Å] -> {out}  ({E_all.size} points)")

if __name__ == "__main__":
    main()
