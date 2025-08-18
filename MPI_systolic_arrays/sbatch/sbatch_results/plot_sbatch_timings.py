import os
import re
import matplotlib.pyplot as plt

root = "."
sizes = ["500", "1000", "2000"]
data = {}  # {size: {np: time}}

for size in sizes:
    folder = "value" + size
    data[size] = {}
    for fname in os.listdir(folder):
        match = re.match(r"timing_N(\d+)_np(\d+)_\d+\.txt", fname)
        if match:
            file_size, np = match.groups()
            np = int(np)
            with open(os.path.join(folder, fname)) as f:
                line = f.readline().strip()
                if "time=" in line:
                    time = float(line.split("time=")[-1].strip())
                    data[size][np] = time

# === Plot one figure per size ===
for size, timings in data.items():
    x = sorted(timings.keys())
    y = [timings[np] for np in x]
    plt.figure()
    plt.plot(x, y, 'o-', label="Size " + size)
    plt.xlabel("Number of Processes")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time for Size " + size)
    plt.grid(True)
    plt.legend()
    plt.savefig("timing_N" + size + ".png")

# === Plot all sizes together ===
plt.figure()
for size, timings in data.items():
    x = sorted(timings.keys())
    y = [timings[np] for np in x]
    plt.plot(x, y, 'o-', label="Size " + size)
plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs NP (All Sizes)")
plt.grid(True)
plt.legend()
plt.savefig("timing_all_sizes.png")
plt.show()

