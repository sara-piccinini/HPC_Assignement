import matplotlib.pyplot as plt
import numpy as np

niter   = [10000, 20000, 50000]
ti_name = ['./data/time_conf_a_', './data/time_conf_b_']
label   = ['configuation a', 'configuration b']
data_th = []
data_tm = []
min_tim = [[], []]

colors = ['r', 'g', 'b']
fig, ax = plt.subplots(3,2) 
fig.suptitle("Performance with dynamic scheduling (100 chucks)", fontsize=15)
fig.supylabel("Time [s]", fontsize=10)
fig.supxlabel("Number of threads", fontsize=10)
fig.subplots_adjust(top=0.89, hspace=0.25, wspace=0.15, left=0.09, right=0.97, bottom=0.09)

for i, name in enumerate(ti_name):
    conf_thread = []
    conf_time   = []

    ax[0][i].set_title(label[i])

    for j, ni in enumerate(niter):

        nti     = [] #num thread for this iterations
        time_i  = [] #time for this iteration
        with open(name + str(ni), 'r') as t:
            for line in t.readlines():
                line = line.split()
                nti.append(int(line[0]))
                time_i.append(float(line[1]))

            if (i == 0):

                if(j == 0):
                    ax[j][i].plot(nti, [time_i[0]/k for k in nti], '--c', label="Ideal")
                else:
                    ax[j][i].plot(nti, [time_i[0]/k for k in nti], '--c')

                ax[j][i].plot(nti, time_i, colors[j], label=str(ni))
            else:
                ax[j][i].plot(nti, time_i, colors[j])
                ax[j][i].plot(nti, [time_i[0]/k for k in nti], '--c')

            min_tim[i].append(nti[time_i.index(min(time_i))])
            ax[j][i].scatter(min_tim[i][j], min(time_i), color=colors[j])

            ax[j][i].set_xlim(xmin=0)
            ax[j][i].set_ylim(ymin=0)            

        conf_thread.append(nti)
        conf_time.append(time_i) 

    data_th.append(conf_thread)  
    data_tm.append(conf_time) 

fig.legend()  
fig.savefig('./plot/Timevsnumth_dinamsched.jpg', dpi=450)
plt.show()  

# compute speed up
for i in range(len(ti_name)):
    for j in range(len(niter)):
        norm = data_tm[i][j][0]
        for k in range(len(data_tm[i][j])):
            data_tm[i][j][k] = (1 - data_tm[i][j][k]/norm)*100

width=1
for i, lb in enumerate(label):

    fig, ax = plt.subplots(1,1)
    fig.suptitle("Speedup with " + lb + ' dynamic scheduling', fontsize=15)
    fig.subplots_adjust(top=0.89, hspace=0.25, wspace=0.15, left=0.09, right=0.97, bottom=0.09)

    for j in range(-1, len(niter)-1):
        bar1 = ax.bar([data_th[i][j+1][k] + j*width for k in range(2, len(data_th[i][j+1]))], data_tm[i][j+1][2:], width, label=str(niter[j+1]))
    
    plt.xticks(data_th[i][j][2:], data_th[i][j][2:])
    ax.set_xlabel('Threads', fontsize=10)
    ax.set_ylabel('% of speedup', fontsize=10)
    ax.set_xlim(1, 54)
    ax.set_ylim(70, 100)
    ax.legend()
    fig.savefig('./plot/speedupscheddinam_'+lb, dpi=450)
    plt.show()

# fig, ax = plt.subplots(1,1)
# fig.suptitle("Speedup with " + lb, fontsize=15)
# fig.subplots_adjust(top=0.89, hspace=0.25, wspace=0.15, left=0.09, right=0.97, bottom=0.09)

# for i, lb in enumerate(label):
#     for j in range(len(niter)):
#         ax.scatter(data_th[i][j], data_tm[i][j])

# plt.show()