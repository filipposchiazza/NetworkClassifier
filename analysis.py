import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from NetworkClassifier import NetworkClassifier
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import umap

#################################################################################################
# Loading real datasets

def load_iris_dataset():
    "Load Iris dataset"
    iris = datasets.load_iris()
    iris_data = iris.data
    iris_data = normalize_dataset(iris_data)
    iris_target = iris.target
    x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, shuffle=True, test_size=0.3)
    return x_train, x_test, y_train, y_test

def load_wine_dataset():
    wine = datasets.load_wine()
    wine_data = wine.data
    wine_data = normalize_dataset(wine_data)
    wine_target = wine.target
    x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, shuffle=True, test_size=0.3)
    return x_train, x_test, y_train, y_test

def load_breast_cancer_dataset():
    breast = datasets.load_breast_cancer()
    breast_data = breast.data
    breast_data = normalize_dataset(breast_data)
    breast_target = breast.target
    x_train, x_test, y_train, y_test = train_test_split(breast_data, breast_target, shuffle=True, test_size=0.3)
    return x_train, x_test, y_train, y_test

def load_digits_dataset():
    digits = datasets.load_digits()
    digits_data = digits.data
    digits_data = normalize_dataset(digits_data)
    digits_target = digits.target
    x_train, x_test, y_train, y_test = train_test_split(digits_data, digits_target, shuffle=True, test_size=0.3)
    return x_train, x_test, y_train, y_test

def normalize_dataset(data):
    "Take as input the dataset and normalize it (use MinMaxScaler)"
    # define the MinMaxScaler
    scaler = MinMaxScaler()
    # transform the data
    data_scaled = scaler.fit_transform(data)
    return data_scaled
#################################################################################################


def plot_toy_dataset(data, targets, title):
    x, y = np.split(data, 2, axis=1)
    plt.scatter(x, y, c=targets, s=3)
    plt.xlabel('$x_1$')
    plt.ylabel('$y_1$')
    plt.title(title)
    plt.show()
    
    
def make_prediction(x_test, y_test, graph, b, alpha, verbose = False):
    prediction = []
    count = 0
    num_test = len(y_test)
    if verbose == True:
        for i in range(num_test):
            _,_,p = graph.predict(sample=x_test[i], b=b, alpha=alpha)
            prediction.append(p)
            count += 1
            print(str(count) + '/' + str(num_test))
    else:
        for i in range(num_test):
            _,_,p = graph.predict(sample=x_test[i], b=b, alpha=alpha)
            prediction.append(p)
    return prediction

    
def perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose = False):
    net = NetworkClassifier()
    accuracy = {}
    count = 0
    for k in K:
        for eps in EPS:
            for b in B:
                for alpha in ALPHA:
                    net.build_graph(data=x_train, targets=y_train, k=k, eps=eps)
                    prediction = make_prediction(x_test, y_test, net, b=b, alpha=alpha)
                    name = str(k) + '-' + str(eps) + '-' + str(b)+ '-' + str(alpha)
                    accuracy[name] = np.sum(prediction==y_test)
                    count += 1
                    if verbose == True:    
                        print(str(count) + '/' + str(len(K)*len(EPS)*len(B)*len(ALPHA)))
    return accuracy


def digit_classification(x_train, x_test, y_train, y_test, k, eps, b, alpha, verbose = False):
    net = NetworkClassifier()
    net.build_graph(data=x_train, targets=y_train, k=k, eps=eps)
    prediction = make_prediction(x_test=x_test, y_test=y_test, graph=net, b=b, alpha=alpha, verbose = verbose)
    accuracy = np.sum(prediction==y_test)
    return accuracy
    
    



if __name__ == '__main__':
    
    # Circle dataset without noise
    data, targets = datasets.make_circles(n_samples=300, noise=0.0, factor=0.8, random_state=1)
    # plotting
    plot_toy_dataset(data=data, targets=targets, title='Circles without noise')
    # divide train and testing
    x_train, x_test, y_train, y_test = train_test_split(data, targets, shuffle=True, test_size=0.3)
    # perform grid search
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.3, 0.6, 0.8, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    object_to_be_saved = {"Circles without noise" : accuracy}
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(object_to_be_saved, f)
    f.close()
    

    # Circle dataset with noise
    data, targets = datasets.make_circles(n_samples=300, noise=0.25, factor=0.8, random_state=1)
    # plotting
    plot_toy_dataset(data=data, targets=targets, title='Circles with noise=0.25')
    # divide train and testing
    x_train, x_test, y_train, y_test = train_test_split(data, targets, shuffle=True, test_size=0.3)
    # perform grid search
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.3, 0.6, 0.8, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Circles with noise = 0.25"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    # Moon dataset without noise
    data, targets = datasets.make_moons(n_samples=400, noise=0.0, random_state=1)
    # plotting
    plot_toy_dataset(data=data, targets=targets, title='Moon dataset without noise')
    # divide train and testing
    x_train, x_test, y_train, y_test = train_test_split(data, targets, shuffle=True, test_size=0.3)
    # perform grid search
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.3, 0.6, 0.8, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Moon without noise"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    # Moon dataset with noise
    data, targets = datasets.make_moons(n_samples=400, noise=0.25, random_state=1)
    # plotting
    plot_toy_dataset(data=data, targets=targets, title='Moon dataset with noise=0.25')
    # divide train and testing
    x_train, x_test, y_train, y_test = train_test_split(data, targets, shuffle=True, test_size=0.3)
    # perform grid search
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.3, 0.6, 0.8, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Moon with noise = 0.25"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    # Iris dataset
    x_train, x_test, y_train, y_test = load_iris_dataset()
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.25, 0.5, 0.75, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Iris"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    # Wine dataset
    x_train, x_test, y_train, y_test = load_wine_dataset()
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.25, 0.5, 0.75, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Wine"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    # Breast cancer dataset
    x_train, x_test, y_train, y_test = load_breast_cancer_dataset()
    K = [3,5,7]
    EPS = [0.2, 0.4, 0.6]
    B = [3,5,7]
    ALPHA = [0.0, 0.25, 0.5, 0.75, 1.0] 
    accuracy = perform_grid_search(x_train, x_test, y_train, y_test, K, EPS, B, ALPHA, verbose=True)
    results["Breast_cancer"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    ###########################################################################
    # test further combinations for Breast cancer dataset
    K = [5,7]
    EPS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    B = 5
    ALPHA =  0.5
    
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    
    count=0
    for k in K:
        for eps in EPS:
            net = NetworkClassifier()
            net.build_graph(x_train, y_train, k=k, eps=eps)
            prediction = make_prediction(x_test, y_test, net, b=B, alpha=ALPHA)
            name = str(k) + '-' + str(eps) + '-' + str(B)+ '-' + str(ALPHA)
            results["Breast_cancer"][name] = np.sum(prediction==y_test)
            count += 1    
            print(str(count) + '/' + str(len(EPS)*len(K)))
            
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
     
       
    K = 10
    EPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    B = 5
    ALPHA =  0.5
    
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    
    count = 0
    for eps in EPS:
        net = NetworkClassifier()
        net.build_graph(x_train, y_train, k=K, eps=eps)
        prediction = make_prediction(x_test, y_test, net, b=B, alpha=ALPHA)
        name = str(K) + '-' + str(eps) + '-' + str(B)+ '-' + str(ALPHA)
        results["Breast_cancer"][name] = np.sum(prediction==y_test)
        count += 1    
        print(str(count) + '/' + str(len(EPS)))
    
    
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
     
    
    
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k3 = []
    k5 = []
    k7 = []
    k10 = []
    breast_accuracy = results["Breast_cancer"]
    for eps in x:
        k3.append(breast_accuracy['3-'+str(eps)+'-5-0.5'])
        k5.append(breast_accuracy['5-'+str(eps)+'-5-0.5'])
        k7.append(breast_accuracy['7-'+str(eps)+'-5-0.5'])
        k10.append(breast_accuracy['10-'+str(eps)+'-5-0.5'])
        
    k3 = np.asarray(k3) / len(y_test) * 100
    k5 = np.asarray(k5) / len(y_test) * 100
    k7 = np.asarray(k7) / len(y_test) * 100
    k10 = np.asarray(k10) / len(y_test) * 100
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(x, k3, c='b', label='k=3')
    ax1.plot(x, k5, c='r', label='k=5')
    ax1.plot(x, k7, c='g', label='k=7')
    ax1.plot(x, k10, c='y', label='k=10')
    plt.xlabel("$\epsilon$")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy with respect to $\epsilon$ for different values of $k$")
    plt.legend(loc='upper right');
    plt.show()
    
    
    
    k = 5 
    eps = 0.2
    B = [3, 5, 7]
    ALPHA = [0.15, 0.4, 0.65, 0.9]
    
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    
    count=0
    for b in B:
        for alpha in ALPHA:
            net = NetworkClassifier()
            net.build_graph(x_train, y_train, k=k, eps=eps)
            prediction = make_prediction(x_test, y_test, net, b=b, alpha=alpha)
            name = str(k) + '-' + str(eps) + '-' + str(b)+ '-' + str(alpha)
            results["Breast_cancer"][name] = np.sum(prediction==y_test)
            count += 1    
            print(str(count) + '/' + str(len(B)*len(ALPHA)))
            
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    k = 5
    eps = 0.2
    b = 15
    ALPHA = [0.0, 0.15, 0.25, 0.4, 0.5, 0.65, 0.75, 0.9, 1.0]
    
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    
    count=0
    for alpha in ALPHA:
        net = NetworkClassifier()
        net.build_graph(x_train, y_train, k=k, eps=eps)
        prediction = make_prediction(x_test, y_test, net, b=b, alpha=alpha)
        name = str(k) + '-' + str(eps) + '-' + str(b)+ '-' + str(alpha)
        results["Breast_cancer"][name] = np.sum(prediction==y_test)
        count += 1    
        print(str(count) + '/' + str(len(ALPHA)))
            
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    x = [0.0, 0.15, 0.25, 0.4, 0.5, 0.65, 0.75, 0.9, 1.0]
    b3 = []
    b5 = []
    b7 = []
    b15 = []
    breast_accuracy = results["Breast_cancer"]
    for alpha in x:
        b3.append(breast_accuracy['5-0.2-3-'+str(alpha)])
        b5.append(breast_accuracy['5-0.2-5-'+str(alpha)])
        b7.append(breast_accuracy['5-0.2-7-'+str(alpha)])
        b15.append(breast_accuracy['5-0.2-15-'+str(alpha)])
    
    b3 = np.asarray(b3) / len(y_test) * 100
    b5 = np.asarray(b5) / len(y_test) * 100
    b7 = np.asarray(b7) / len(y_test) * 100
    b15 = np.asarray(b15) / len(y_test) * 100
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(x, b3, c='b', label='b=3')
    ax1.plot(x, b5, c='r', label='b=5')
    ax1.plot(x, b7, c='g', label='b=7')
    ax1.plot(x, b15, c='y', label='b=15')
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Accuracy(%)")
    plt.title("Accuracy with respect to " + r"$\alpha$ for different values of $b$")
    plt.legend(loc='lower left');
    plt.show()
    
    
    ###########################################################################
    
    
    
    
    # Digit dataset
    x_train, x_test, y_train, y_test = load_digits_dataset()
    k = 10
    eps = 0.0
    b = 30
    alpha = 0.2
    accuracy = digit_classification(x_train, x_test, y_train, y_test, k=k, eps=eps, b=b, alpha=alpha, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Digits"][str(k) + "-" + str(eps) + "-" + str(b) + "-" +str(alpha)] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    # Use Umap with Digit dataset 
    digits = datasets.load_digits()
    digits_data = digits.data
    digits_target = digits.target
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=1)
    reducer.fit(digits_data)
    embedding = reducer.embedding_
    digits_data_reduced = normalize_dataset(embedding)
    x_train, x_test, y_train, y_test = train_test_split(digits_data_reduced, digits_target, shuffle=False, test_size=0.3)
    k = 5
    eps = 0.0
    b = 5
    alpha = 0.0
    accuracy = digit_classification(x_train, x_test, y_train, y_test, k=k, eps=eps, b=b, alpha=alpha, verbose=True)
    with open("grid_search_results.pkl", 'rb') as f:
        results = pkl.load(f)
    f.close()
    results["Digits with Umap"] = accuracy
    with open("grid_search_results.pkl", 'wb') as f:
        pkl.dump(results, f)
    f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    

    