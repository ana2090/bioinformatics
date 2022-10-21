import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean, cityblock
import matplotlib.pyplot as plt

class kNN:
    def __init__(self, k, labels, distance_formula="euclidean"):
        self.k = k
        self.labels = labels
        self.distance = distance_formula
    
    ######### SCALING #########
    
    '''
    normalize X by taking the mean and std of each column
    '''
    @staticmethod
    def normalize(X):
        for i in range(X.shape[1]): # i is the feature index
            new_col = X[:, i] - X[:, i].mean() / X[:, i].std()
            X[:, i] = new_col
        return X

    '''
        scale by dividing the max of each feature from that column
    '''
    @staticmethod
    def max_scale(X):
        for i in range(X.shape[1]):
            max_col = max(X[:, i])
            X[:, i] /= max_col
        return X
    
    ######## ALGORITHM ########
    
    '''
        similarity measure between sample p and sample q
        
        3 options:
            cosine, euclidean, cityblock
            given in object creation
    '''
    def dist(self, p, q):
        if self.distance == "cosine":
            return cosine(p, q)
        if self.distance == "euclidean":
            return euclidean(p, q)
        if self.distance == "cityblock":
            return cityblock(p, q)
        quit("\nEXIT: invalid distance measure function\n")
        
    '''
        If outputs of k samples match more than k/2 -> prediction correct
        When ties, randomly select
        
        For each sample q, if the vote of q matches the vote of at least half of its k closest neighbors, then the prediction is correct


    '''
    def get_vote(self, k_samples):
        votes = {}
        for label in self.labels:
            votes.setdefault(label, len([i for i in k_samples if i[-1] == label]))
        return votes
    
    '''
        get the k samples from other nearest to q
    '''
    def get_nearest(self, q, other):
        distances = []
        
        for i in other:
            distances.append((i, self.dist(q[:-1], i[:-1])))
            
        distances = sorted(distances, key=lambda x:x[1])
        return np.array([i[0] for i in distances][:self.k])
        
    '''
        k nearest neighbors algorithm
    '''
    def kNN(self, X):
        predict = 0
        half_k = self.k / 2.0
        
        for j in range(X.shape[0]):
            q = X[j]
            
            X_q = np.delete(X, j, 0)
            
            k_samples = self.get_nearest(q, X_q)

            vote = self.get_vote(k_samples)
            
            if vote[X[j, -1]] >= half_k:
                predict += 1
                
        return predict / X.shape[0]
    
    
if __name__ == "__main__":
    # different ways to measure the distance between 2 samples
    dists = ["cosine", "euclidean", "cityblock"]
    
    ############   HW6 DATASET  ##############
    df = pd.read_csv("data/hw6_data.csv")
    y = df["Outcome"].to_numpy()  # labels
    X = df.drop(columns=["Outcome"]).to_numpy()  # features
    
    # Graph of the 1st Dataset's Raw Data
    '''
    color = np.array(["orange", "pink"])
    labels = np.unique(y)
    plt_y = []
    for i in y:
        plt_y.append(color[np.where(labels == i)][0])
    plt.title("Dataset 1 Raw Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.scatter(X[:, 0], X[:, 1], c=plt_y)
    plt.show()'''

    print("HW6 dataset")
    for k in [3, 5, 7]:
        clstr = kNN(k, np.unique(y), dists[1])
        X = clstr.normalize(X)
        X_y = np.c_[X, y]
        out = clstr.kNN(X_y)
        print(f"k = {k}, Accuracy {out}")
    print()
        
    ###########  IRIS DATASET ##############
    df = pd.read_csv("data/iris.data", header=None)
    df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
    
    # remove sepal and pedal widths
    df = df.drop(columns=["sepal_wid", "petal_wid"])
    
    y = df["class"].to_numpy() # labels
    X = df.drop(columns=["class"]).to_numpy()  # features
    

    # Graph of Raw Data for the Iris data
    '''
    color = np.array(["g", "b", "red"])
    labels = np.unique(y)
    plt_y = []
    for i in y:
        plt_y.append(color[np.where(labels == i)][0])
    plt.title("Dataset 2 Raw Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.scatter(X[:, 0], X[:, 1], c=plt_y)
    plt.show()'''
    
    print("Iris Dataset")
    for k in [3, 5, 7]:
        clstr = kNN(k, np.unique(y), dists[1])
        X = clstr.normalize(X)
        X_y = np.c_[X, y]
        out = clstr.kNN(X_y)
        print(f"k = {k}, Accuracy {out}")
    print()
