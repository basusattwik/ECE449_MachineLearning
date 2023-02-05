import torch
import numpy as np
import hw1_utils

# Distance measure
def get_distance(x, y):
    return torch.norm(x-y)**2


# Kmeans implementation
def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = hw1_utils.load_data()

    maxItr = 10
    c = init_c
    K = 2

    D = X.size(dim=0)
    T = X.size(dim=1) # total number of data points
    r = torch.zeros([K, T]) # keeps track of assignments
    d = torch.zeros([K, T]) # keeps track of distances of points from centroids

    bPlot = False
    i = 0       
    while i < maxItr:

        # get distance of points from cluster centers
        for n in range(T):
            for k in range(K):
                # Calculate distance between each point and the centroids
                d[k, n] = get_distance(X[:,n], c[:,k])

        
        # Assign points to clusters
        r[r!=0] = 0 # Reset
        for n in range(T):
            ind = torch.argmin(d[:,n])
            for k in range(K):           
                if k == ind:
                    r[k,n] = 1


        # update centroids       
        for k in range(K):
            c[:,k] = torch.sum(r[k,:] *  X, 1) / torch.sum(r[k,:])

        
        # visualize clusters
        if bPlot:
            c1 = torch.reshape(c[:,0], (D,1))
            c2 = torch.reshape(c[:,1], (D,1))
            x1 = X[:,torch.argwhere(r[0,:])]
            x2 = X[:,torch.argwhere(r[1,:])]
            hw1_utils.vis_cluster(c1, x1, c2, x2)

        # Update loop index
        i += 1

    return c

# main entrypoint
centroids = k_means()
print('--- Final Centroids:', centroids)
print('Done!')
