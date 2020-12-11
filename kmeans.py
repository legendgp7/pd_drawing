import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

size = 30



def distEuclid(x, y):
    return np.sqrt(np.sum((x - y) ** 2))



def genDataset(n, dim):
    data = []
    while len(data) < n:
        p = np.around(np.random.rand(dim) * size, decimals=2)
        data.append(p)
    return data



def initCentroid(data, k):
    num, dim = data.shape
    centpoint = np.zeros((k, dim))
    l = [x for x in range(num)]
    np.random.shuffle(l)
    for i in range(k):
        index = int(l[i])
        centpoint[i] = data[index]
    return centpoint



def KMeans(data, k):

    num = np.shape(data)[0]


    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1


    change = True

    cp = initCentroid(data, k)

    while change:
        change = False


        for i in range(num):
            minDist = 9999.9
            minIndex = -1


            for j in range(k):
                dis = distEuclid(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j


            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist


        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x, 0] == j]]
            cp[j] = np.mean(pointincluster, axis=0)


    return cp, cluster



def Show(data, k, cp, cluster):
    num, dim = data.shape
    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

    if dim == 2:
        for i in range(num):
            mark = int(cluster[i, 0])
            plt.plot(data[i, 0], data[i, 1], color[mark] + 'o')

        for i in range(k):
            plt.plot(cp[i, 0], cp[i, 1], color[i] + 'x')

    elif dim == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(num):
            mark = int(cluster[i, 0])
            ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=color[mark])

        for i in range(k):
            ax.scatter(cp[i, 0], cp[i, 1], cp[i, 2], c=color[i], marker='x')

    plt.show()




