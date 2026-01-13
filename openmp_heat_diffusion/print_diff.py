import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def init_matrix():
    global axi, cbar
    if cbar is not None:
        cbar.remove()
        ax.set_title('Heat diffusion config b')
    axi = ax.imshow(matrix[0], cmap='rainbow', vmin=matrix.min(), vmax=matrix.max())
    cbar = fig.colorbar(axi, ax=ax)
    return [axi]

def animate(i):  
    axi.set_data(matrix[i])
    return [axi]

def num_matrix():
    i = 0
    with open(temp+'a', 'r') as t:
            row = t.readline()
            while(row):
                row = t.readline()
                i += 1
    return i//1024

def cond(nmi):
    if(nmi == 1):
        return [0, 0]
    elif(nmi == 3):
        return [0, 1]
    elif(nmi == 7):
        return [1, 0]
    elif(nmi == 11):
        return [1, 1]
    else:
        return [-1, -1]

temp = './data/temp_conf_'
confs = ['a', 'b']

nm = num_matrix()
matrix = np.zeros((nm, 1024, 1024))

## for gifs
# fig, ax = plt.subplots()
# ax.set(title=('Heat diffusion config a'), xlim=([0, 1024]))
cbar = None

##for partial matrix
figm, axm = plt.subplots(2, 2)

for c in confs:
    with open(temp+c, 'r') as t:
        nmi = 0
        figm.suptitle('Heat diffusion config ' + c)
        row = t.readline()
        while(row):

            for i in range(1024):
                row = row.split()
                for j in range(1024):
                    matrix[nmi][i][j] = (float(row[j]))
                row = t.readline()

            nmi += 1

            f = cond(nmi)
            if(f[0] != -1):
                axmi = axm[f[0]][f[1]].imshow(matrix[nmi-1], cmap='rainbow', vmin=matrix.min(), vmax=matrix.max())            
                if(f == [0, 0]):
                    if(cbar is not None):
                        cbar.update_normal(axmi)
                    else:
                        cbar = figm.colorbar(axmi, ax=axm)  
                figm.savefig('./plot/hd_conf_' + c + '.jpg')

            row = t.readline()

        # anim = animation.FuncAnimation(fig, animate, init_func=init_matrix, frames=(nm), interval=100, blit=False)
        # anim.save('./plot/temp_' + c + '.gif')




