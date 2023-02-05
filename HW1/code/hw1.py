import torch
import numpy as np
import hw1_utils

# Distance measure
def get_distance(x, y):
    return torch.norm(x-y)**2

# Assign every point in the dataset to one of the clusters
def get_assigmnents():
    pass

# Update centroids to new positions
def update_centroids():
    pass

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

    D = X.size(dim=0)
    T = X.size(dim=1) # total number of data points
    print('--- T = ', T)
    K = 2
    r = torch.zeros([K, T]) # keeps track of assignments
    d = torch.zeros([K, T]) # keeps track of distances of points from centroids

    maxItr = 50
    i = 0
    c = init_c

    bPlot = False

    print('--- size of X: ', X.size())
    print('--- X = ', X)
    print('--- size of c init: ', c.size())
    print('--- c init = ', c)
    print('--- size of r init: ', r.size())
    print('--- rinit = ', r)
    print('--- size of c: ', c.size())
    print('--- Step ', i, 'c = ', c)

    while i < maxItr:

        # get distance of points from cluster centers
        for n in range(T):
            for k in range(K):
                # Calculate distance between each point and the centroids
                d[k, n] = get_distance(X[:,n], c[:,k])

        print('--- size of d:', d.size())
        print('--- dist, d = ', d)
        
        # Assign points to clusters
        for n in range(T):
            ind = torch.argmin(d[:,n])
            for k in range(K):           
                if k == ind:
                    r[k,n] = 1

        print('--- size of r: ', r.size())
        print('--- r = ', r)

        # update centroids       
        for k in range(K):
            c[:,k] = torch.sum(r[k,:] *  X, 1) / torch.sum(r[k,:])

        print('--- size of c: ', c.size())
        print('--- Step ', i+1, 'c = ', c)
        
        # visualize clusters
        if bPlot:
            c1 = torch.reshape(c[:,0], (D,1))
            c2 = torch.reshape(c[:,1], (D,1))
            r0 = r[0,:]
            r1 = r[1,:]
            ind0 = torch.argwhere(r0)
            ind1 = torch.argwhere(r1)
            x1 = X[:,ind0]
            x2 = X[:,ind1]
            hw1_utils.vis_cluster(c1, x1, c2, x2)

        # Reset
        r[r!=0] = 0

        # Update loop index
        i += 1

    return c, r

# main entrypoint
cent, r = k_means()

print('--- Final Centroids:', cent)
print('Done!')
