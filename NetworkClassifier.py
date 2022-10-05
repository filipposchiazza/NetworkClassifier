import numpy as np
import networkx as nx
from scipy.spatial import distance


class NetworkClassifier:

    def __init__(self):
        self.k = None 
        self.eps = None
        self.b = None
        self.alpha = None
        self.data_sorted = None
        self.targets_sorted = None
        self.num_nodes = None
        self.node_labels = None
        self.joint_graph = nx.DiGraph()
        self.disjoint_graph = nx.DiGraph()
        self.distance_matrix = None
        self.adjacency_matrix = None
        self.key_list = None
        self.graph_dict = {} 
        
    #############################################################################################
    # TRAINING PHASE: GRAPH BUILDING
    def build_graph(self, data, targets, k, eps):
        self.k = k
        self.eps = eps        
        self.data_sorted, self.targets_sorted = self._sort_dataset_by_targets(data=data, targets=targets)
        self._add_nodes_with_attribute(data=self.data_sorted, targets=self.targets_sorted)
        self.distance_matrix = self._calculate_distance(data=self.data_sorted)
        self._add_edges()
        self._remove_edges_different_labels()
        self._separate_independent_graphs(targets)
        
        
    def _sort_dataset_by_targets(self, data, targets):
        "Sort the dataset and the targets according to target values"
        idx = np.argsort(targets)
        data_sorted = data[idx]
        targets_sorted = targets[idx]
        return data_sorted, targets_sorted
    
    def _add_nodes_with_attribute(self, data, targets):
        "Each row-data is added as a node with the target as the node attribute"
        self.num_nodes = len(data)
        self.node_labels = np.arange(self.num_nodes)
        self.joint_graph.add_nodes_from(self.node_labels)
        dic_for_attributes = {m:n for (m,n) in zip(self.node_labels, targets)}
        nx.set_node_attributes(self.joint_graph, dic_for_attributes, "target")
        
    def _calculate_distance(self, data):
        "Evaluate the distance between data-points and return a distance matrix"
        distance_matrix = distance.cdist(data, data, metric = 'euclidean')
        return distance_matrix
    
    def _add_edges(self):
        "Add the edges according to eps-criteria and kNN criteria"
        for j in self.node_labels:
            support = self.distance_matrix[j] < self.eps
            support[j] = False # remove self link count
            density = np.sum(support) 
            if density > self.k:
                eps_indexes = np.where(self.distance_matrix[j] < self.eps)[0]
                edges_list = [(j, n) for n in eps_indexes]  
            else:
                knn_indexes = np.argpartition(self.distance_matrix[j], kth=self.k)
                knn_indexes = knn_indexes[:self.k+1]
                edges_list = [(j, n) for n in knn_indexes]
            self.joint_graph.add_edges_from(edges_list)
        self.joint_graph.remove_edges_from(nx.selfloop_edges(self.joint_graph))
        
    def _remove_edges_different_labels(self):
        "Remove the edges between nodes with different target attribute"
        self.disjoint_graph = self.joint_graph.copy()
        for i in self.node_labels:
            for j in self.node_labels:
                if self.disjoint_graph.has_edge(i, j) and self.disjoint_graph.nodes[i]['target'] != self.disjoint_graph.nodes[j]['target']:
                    self.disjoint_graph.remove_edge(i, j)
        self.adjacency_matrix = nx.to_numpy_array(self.disjoint_graph)
        
    def _separate_independent_graphs(self, targets):
        self.key_list = np.unique(targets)
        self.graph_dict = {key : None for key in self.key_list}
        # Obtain one graph component for each class label
        for j in self.key_list:
            node_list_subgraph_j = [x for x,y in self.joint_graph.nodes(data=True) if y['target']==j]
            self.graph_dict[j] = self.disjoint_graph.subgraph(node_list_subgraph_j).copy()
        """
        # Obtain one graph component for each class label
        components = [self.disjoint_graph.subgraph(c).copy() for c in nx.weakly_connected_components(self.disjoint_graph)]
        # Link each subgraph to the corresponding label in the graph dictionary
        for j in range(len(components)):
            node_attributes = nx.get_node_attributes(components[j], name='target')
            key = list(node_attributes.values())[0]
            self.graph_dict[key] = components[j]
        """   
    
    #############################################################################################
    # TESTING PHASE: TEST ISTANCES CLASSIFICATION
    
    def predict(self, sample, b, alpha):
        self.b = b
        self.alpha = alpha
        sample_distances = self._sample_distance(sample)
        sample_neighbours_idx = self._find_sample_neighbours(sample_distances)
        graph_dict_updated = self._update_graph_dictionary(sample_neighbours_idx)
        # betweenness is a dictionary of dictionary 
        # the external one related to each subgraph and the internal to each node
        betweeness = self._evaluate_betweenness_centrality(graph_dict_updated)
        # obtain a dictionary with the value of the b-lower difference average for each class
        betweeness_differences = self._evaluate_average_betweeness_differences(betweeness)
        # evaluate the number of link of the inserted node for each class (dictionary)
        sample_degrees = self._evaluate_sample_degrees(graph_dict_updated)
        # evaluate probabilities (dictionary)
        probabilities = self._evaluate_probabilities(betweeness_differences, sample_degrees)
        prediction = self._predict(probabilities)
        
        return graph_dict_updated, probabilities, prediction
        
        
    def _sample_distance(self, sample):
        "Calculate the distances between the sample and the training dataset"
        sample = sample.reshape((1,-1))
        sample_distances = distance.cdist(sample, self.data_sorted)
        sample_distances = sample_distances.reshape(sample_distances.size, )
        return sample_distances
    
    def _find_sample_neighbours(self, sample_distances):
        "Find the indexes of the neighbours to be linked with the sample according to kNN and epsilon criteria"
        support = sample_distances <= self.eps
        density = np.sum(support)
        if density > self.k:
            neighbours_idx = np.where(sample_distances <= self.eps)[0]
        else:
            knn_indexes = np.argpartition(sample_distances, kth=self.k)
            neighbours_idx = knn_indexes[:self.k]
        return neighbours_idx
    
    def _update_graph_dictionary(self, sample_neighbours_idx):
        graph_dict_updated = {key : None for key in self.key_list}
        for j in self.key_list:
            #copy the graph
            graph_dict_updated[j] = self.graph_dict[j].copy()
            #add a new node 
            node_id = list(self.graph_dict[j].nodes)[-1] + 1
            graph_dict_updated[j].add_node(node_id)
            #add the edges
            graph_node_names = list(self.graph_dict[j].nodes)
            support = np.isin(sample_neighbours_idx, graph_node_names)
            sample_neighbours_idx = np.asarray(sample_neighbours_idx)
            nodes_to_connect = sample_neighbours_idx[support] 
            edge_list = [(node_id, n) for n in nodes_to_connect]
            graph_dict_updated[j].add_edges_from(edge_list)
            edge_list_reverse = [(n, node_id) for n in nodes_to_connect]
            graph_dict_updated[j].add_edges_from(edge_list_reverse)
            #graph_dict_updated[j] = graph_dict_updated[j].to_undirected()
        return graph_dict_updated
    
    def _evaluate_betweenness_centrality(self, graph_dict_updated):
        betweeness = {key : None for key in self.key_list}
        for j in self.key_list:
            betweeness[j] = nx.betweenness_centrality(graph_dict_updated[j], normalized=False)
        return betweeness
    
    def _evaluate_average_betweeness_differences(self, betweeness):
        betweeness_differences = {key : None for key in self.key_list}
        for j in self.key_list:
            betw_list = list(betweeness[j].values())
            betw_list = np.asarray(betw_list)
            sample_betw = betw_list[-1]
            differences = abs(sample_betw - betw_list)
            differences.sort()
            b_average = np.average(differences[:self.b])
            betweeness_differences[j] = b_average
        return betweeness_differences
    
    def _evaluate_sample_degrees(self, graph_dict_updated):
        sample_degrees = {key : None for key in self.key_list}
        for j in self.key_list:
            sample_node = list(graph_dict_updated[j].nodes)[-1]
            degree = graph_dict_updated[j].degree[sample_node]
            sample_degrees[j] = degree
        return sample_degrees
    
    def _evaluate_probabilities(self, betweeness_differences, sample_degrees):
        # Rescale betweeness_differences
        for j in self.key_list:
            betweeness_differences[j] = 1 - betweeness_differences[j]

        probabilities = {key : None for key in self.key_list}
        for j in self.key_list:
            b = betweeness_differences[j] / np.sum(list(betweeness_differences.values()))
            l = sample_degrees[j] / np.sum(list(sample_degrees.values()))
            probabilities[j] = self.alpha * b + (1 - self.alpha) * l
            
        # Normalize probabilities
        for j in self.key_list:
            probabilities[j] = probabilities[j] / np.sum(list(probabilities.values()))
            
        return probabilities
    
    def _predict(self, probabilities):
        prediction = max(probabilities, key=probabilities.get)
        return prediction
            
        
        
        
        
