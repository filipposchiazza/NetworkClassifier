import NetworkClassifier 
import numpy as np
from scipy.spatial import distance


np.random.seed(1)

data = np.random.rand(6, 4)
distance_matrix = distance.cdist(data, data, metric = 'euclidean')
k=2
eps=1
expected_adj_matrix = np.array([[0, 0, 1, 1, 0, 1],
                                [0, 0, 1, 1, 1, 0],
                                [1, 1, 0, 1, 1, 0],
                                [1, 1, 1, 0, 1, 1],
                                [0, 1, 1, 1, 0, 0],
                                [1, 0, 0, 1, 0, 0]])
nc = NetworkClassifier.NetworkClassifier()
nc.train(x_train=data, y_train=0, eps=eps, k=k)
assert np.all(nc.adjacency_matrix == expected_adj_matrix)