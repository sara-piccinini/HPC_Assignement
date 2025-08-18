import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("timings/results.csv")

# Ensure output directory
os.makedirs("timings/plots", exist_ok=True)

# One plot per N
for n_val, group in df.groupby("N"):
    group_sorted = group.sort_values("np")
    plt.figure()
    plt.plot(group_sorted["np"], group_sorted["time"], marker='o')
    plt.xlabel("Number of MPI Processes (np)")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time for N={}".format(n_val))
    plt.grid(True)
    plt.xscale("log", base=2)
    plt.savefig("timings/plots/time_vs_np_N{}.png".format(n_val), dpi=300)
    plt.close()

# Combined plot for all Ns
plt.figure(figsize=(10, 6))
for n_val, group in df.groupby("N"):
    group_sorted = group.sort_values("np")
    plt.plot(group_sorted["np"], group_sorted["time"], marker='o', label="N={}".format(n_val))

plt.xlabel("Number of MPI Processes (np)")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs MPI Processes (All Matrix Sizes)")
plt.xscale("log", base=2)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("timings/plots/time_vs_np_all.png", dpi=300)
plt.show()

