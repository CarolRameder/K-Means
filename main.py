import math
import numpy as np

# the hyperparameter for K-Means Algorithm
k = 3

# hardcoded problem data
inst = np.array([
    [0.25, 0.50],
    [0.50, 0.35],
    [0.25, 0.50],
    [1.00, 0.35],
    [1.40, 0.70],
    [0.50, 0.85],
    [0.25, 1.00],
    [0.75, 1.00],
    [0.35, 1.25],
    [0.85, 1.25],
    [3.25, 0.50],
    [3.50, 0.35],
    [3.00, 1.00],
    [3.25, 0.85],
    [3.45, 0.85],
    [3.75, 0.85],
    [3.25, 1.10],
    [3.00, 3.25],
    [3.25, 3.00],
    [3.10, 3.50],
    [1.00, 2.50],
    [1.20, 2.40],
    [1.25, 2.50],
    [1.50, 2.50],
    [0.65, 2.75],
    [1.20, 2.75],
    [1.37, 2.75],
    [1.00, 3.00],
    [1.10, 3.20],
    [0.85, 3.35],
])


def dist(p1, p2):
    return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def column(matrix, i):
    return [row[i] for row in matrix]


def getmincol(mat, col):
    poz = 0
    minim = np.inf
    for i in range(mat.shape[0]):
        if minim > mat[i][col]:
            minim = mat[i][col]
            poz = i
    return poz


# the algorithm--------------------------------------------------------------------------------------------------------------------------------------------------------

# the centroids are randomly generating, acording to given instances of training
centr = np.random.randn(k, 2)
newcentr = np.zeros((k, 3))

#the clusters are computed as the new iteration begins
newclus = [0] * inst.shape[0]
D = np.zeros((k, inst.shape[0]))
old_clus = [-1] * inst.shape[0]
clus = newclus
old_centr = np.random.randn(k, 2)

while old_clus != clus and not np.array_equal(old_centr,centr):
    newcentr = np.zeros((k, 3))
    # computing distance matrix
    for i in range(k):
        for j in range(inst.shape[0]):
            D[i][j] = dist(inst[j], centr[i])

    # computing the clusters with euclidian distance

    old_clus = clus.copy()
    for i in range(inst.shape[0]):
        clus[i] = getmincol(D, i)
        newcentr[clus[i]][2] =+ 1
        newcentr[clus[i]][0] =+ inst[i][0]
        newcentr[clus[i]][1] =+ inst[i][1]

    old_centr = centr.copy()
    # the centroid is relocated in the middle of the cluster
    for i in range(k):
        if newcentr[i][2] != 0:
            centr[i][0] = newcentr[i][0] / newcentr[i][2]
            centr[i][1] = newcentr[i][1] / newcentr[i][2]

    print(centr)
    print(clus)