import matplotlib.pyplot as plt
import pandas as pd

def parse_timings(filename):
    data = {'size': [], 'np': [], 'time': []}
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            size = int(parts[0].split('=')[1])
            np = int(parts[1].split('=')[1])
            time = float(parts[2].split('=')[1])
            data['size'].append(size)
            data['np'].append(np)
            data['time'].append(time)
    return pd.DataFrame(data)

def plot_per_size(df):
    for size in df['size'].unique():
        sub = df[df['size'] == size].sort_values('np')
        plt.figure()
        plt.plot(sub['np'], sub['time'], marker='o')
        plt.title("Execution time vs processes (Size {}x{})".format(size, size))
        plt.xlabel("Number of Processes")
        plt.ylabel("Time [s]")
        plt.grid(True)
        plt.savefig("timing_{}.png".format(size))

def plot_combined(df):
    plt.figure()
    for size in df['size'].unique():
        sub = df[df['size'] == size].sort_values('np')
        plt.plot(sub['np'], sub['time'], marker='o', label="{}x{}".format(size, size))
    plt.title("Execution Time vs Number of Processes")
    plt.xlabel("Number of Processes")
    plt.ylabel("Time [s]")
    plt.legend()
    plt.grid(True)
    plt.savefig("timing_comparison.png")

if __name__ == "__main__":
    df = parse_timings("timing_results.txt")
    plot_per_size(df)
    plot_combined(df)
    print("Confirmed: The plots were saved as .png files")


