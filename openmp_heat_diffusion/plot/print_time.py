import matplotlib.pyplot as plt
import numpy as np

ti_name = ['./data/time_conf_a', './data/time_conf_b']
label   = ['Conf a', 'Conf b']
num_th  = [[], []]
time    = [[], []]

for i, name in enumerate(ti_name):
    with open(name, 'r') as t:
        for line in t.readlines():
            line = line.split()
            num_th[i].append(int(line[0]))
            time[i].append(float(line[1]))
        plt.plot(num_th[i], time[i])
        plt.xlabel("Number of threads")
        plt.ylabel("Time in second [s]")
        plt.ylim([0,100])
        plt.show()

fig, ax = plt.subplots(1,1)
width=0.4

bar1 = ax.bar([num_th[1][i] - width/2 for i in range(len(num_th[1]))], time[0], width, label=label[0])
bar2 = ax.bar([num_th[1][i] + width/2 for i in range(len(num_th[1]))], time[1], width, label=label[1])

ax.set_title('Comparison between the two configurations')
ax.set_xlabel('Threads')
ax.set_ylabel('Time (s)')
ax.legend()
plt.show()
