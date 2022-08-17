import numpy as np
import networkx as nx
from NetworkClassifier import NetworkClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_dataset():
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_data = normalize_dataset(iris_data)
    iris_target = iris.target
    x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, shuffle=True, test_size=0.3)
    return x_train, x_test, y_train, y_test


def normalize_dataset(data):
    "Take as input the dataset and normalize it (use MinMaxScaler)"
    # define the MinMaxScaler
    scaler = MinMaxScaler()
    # transform the data
    data_scaled = scaler.fit_transform(data)
    return data_scaled

    



if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset()
    netclass = NetworkClassifier()
    netclass.build_graph(x_train=x_train, y_train=y_train, eps=0.1, k=5)
    
    

