import numpy as np
import networkx as nx
from scipy.spatial import distance




class NetworkClassifier:
    
    def __init__(self):
        self.num_nodes = None
        self.density_vector = None
        self.distance_matrix = None
        self.adjacency_matrix = None
    
    def _calculate_distance(self, data):
        "Evaluate the distance between data-points and return a distance matrix"
        distance_matrix = distance.cdist(data, data, metric = 'euclidean')
        self.distance_matrix = distance_matrix
        
    
    def _create_adjacency_matrix_from_distance_matrix(self, distance_matrix, k, eps):
        """Create an adjacency matrix starting from a distance matrix
        
        Parameters
        ----------
        distance_matrix : 2D array
            Distance matrix.
        k : int
            Number of nearest neighbours to link.
        eps : float
            Radius of the epsilon-criteria.

        Returns
        -------
        None.

        """
        # compute the number of nodes
        self.num_nodes = len(distance_matrix) 
        # create an adjacency matrix with all zeros
        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes))
        # create a density vector to distinguish between the method to use (kNN or eps)
        support = distance_matrix <= eps
        self.density_vector = support.sum(axis=1)
        
        for row_index in range(len(distance_matrix)):
            if self.density_vector[row_index] >= k:
                adjacency_matrix[row_index] = self._epsilon_criteria(vector=distance_matrix[row_index], eps=eps)
            else:
                adjacency_matrix[row_index] = self._kNN_criteria(vector=distance_matrix[row_index], k=k)
        
        self._null_diagonal(adjacency_matrix)
        self.adjacency_matrix = adjacency_matrix


    def _epsilon_criteria(self, vector, eps):
        row_adj_matrix = np.where(vector <= eps, 1, 0)
        return row_adj_matrix
    
    def _kNN_criteria(self, vector, k):
        row_adj_matrix = np.zeros(len(vector))
        knn_indexes = np.argpartition(vector, kth=k)
        for j in range(k+1):
            row_adj_matrix[knn_indexes[j]] = 1
        return row_adj_matrix
    
    def _null_diagonal(self, matrix):
        "Set the values on the principal diagonal to zero"
        for i in range(len(matrix)):
            matrix[i][i] = 0
    
    
    def _partition(self, data, targets):
        target_list = np.unique(targets)
        partition = {}
        for i in target_list:
            partition[str(i)] = []
        for j in range(len(targets)):
            partition[str(targets[j])].append(data[j])
        return partition
            
            
    def build_graph(self, x_train, y_train, eps, k):
        self._calculate_distance(data=x_train)
        self._create_adjacency_matrix_from_distance_matrix(distance_matrix=self.distance_matrix, k=k, eps=eps)
        digraph = nx.DiGraph(self.adjacency_matrix)
        
        
        
        
        