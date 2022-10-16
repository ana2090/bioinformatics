'''
    Ana Dobrianova
    BioInformatics
    Assignment 5: Classifiers
    March 30th, 2022
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, file):
        # shuffle the data, rename the columns
        df = self.shuffle_df(pd.read_csv(file, header=None))
        df.columns = ["sepal_len", "sepal_wid", "petal_len", "petal_wid", "class"]
        
        # get rid of the Setosa data
        df = df.loc[df["class"] != "Iris-setosa"]
        
        # normalize the data so that each feature is within the same range
        # also drop the columns of features we don't want
        self.df = self.normalize(df.drop(columns=["sepal_wid", "petal_wid"]))
        self.features = ["sepal_len", "petal_len"]
        
        # virginica is class 0/-1 and versicolor is 1 (so versicolor is the positive class)
        self.classes = ("Iris-virginica", "Iris-versicolor")
        
        # get the X (features) and y (labels) from the prepared dataframe
        self.X, self.y = self.features_labels(self.df)
        
        # create a version of X with bias values (1s in the 1st column)
        self.X_b = np.c_[np.ones((self.X.shape[0], 1)), self.X]
        
        # 0 instead of -1 for y
        self.y_2 = np.where(self.y == -1, 0, 1)
        
    
    ####### DATA PREP FUNCTIONS ###########
    '''
        randomly shuffle around the dataframe
    '''
    def shuffle_df(self, df):
        np.random.seed(23)  # so I can replicate results
        np_df = df.to_numpy()
        np.random.shuffle(np_df)
        return pd.DataFrame(np_df)

    '''
        normalize the features 
    '''
    def normalize(self, df):
        new_df = {}
        for col in df.columns[:-1]:
            mean = np.mean(df[col])
            std = np.std(df[col])
                
            scaled = (df[col] - mean) / std
            new_df.setdefault(col, scaled)
        new_df.setdefault(df.columns[-1], df[df.columns[-1]].to_numpy())
        return pd.DataFrame(new_df)
        
    '''
        get features and labels from dataframe
    '''
    def features_labels(self, df):
        y = np.where(df["class"].to_numpy() == "Iris-versicolor", 1, -1)
        X = df.drop(columns=["class"]).to_numpy(np.float64)
        return X, np.array(y, np.float64)
    
    ##### GENERAL USE FUNCTIONS #############
    def get_y(self):
        return self.y
    
    '''
        basic logistic/sigmoid function
        1 / 1 + e**-t
    '''
    def logistic(self, t):
        return 1.0 / (1.0 + np.exp(-t))
    
    '''
        performs the dot product of the bias and the weights passed
    '''
    def bias_dot(self, weights):
        return np.dot(self.X_b, weights)
    
    '''
        predict classes from input features
    '''
    def predict(self, weights, bias=True):
        if bias:
            return np.where(self.bias_dot(weights) >= 0.0, 1, -1)
        else:
            return np.where(np.dot(self.X, weights) >= 0.0, 1, -1)
    
    '''
        checks the accuracy
    '''
    def accuracy(self, predicted):
        return (predicted == self.y).sum() / len(predicted)
    
    '''
        The following 2 functions are the same as the above 2, but they use the label
        0 instead of -1.
    '''
    def log_predict(self, weights, threshold):
        return np.where(self.logistic(self.bias_dot(weights)) >= threshold, 1, 0)
    
    def log_accuracy(self, predicted):
        return (predicted == self.y_2).sum() / len(predicted)
        
    '''
        t/f for class
    '''
    def check_pred(self, y_pred, y_true):
        if y_true == 1 and y_pred >= 0:
            return True
        if y_true == -1 and y_pred < 0:
            return True 
        return False
    
    '''
        plot samples

    '''
    def plot_samples(self, weights):
        # the points feature points for each class
        pos = np.array([self.X[i] for i in range(len(self.y)) if self.y[i] == 1])
        neg = np.array([self.X[i] for i in range(len(self.y)) if self.y[i] == -1])
        
        # the decision boundary
        x = np.array([min(self.X[:, 0]), max(self.X[:, 1])])
        y = (-weights[1] / weights[2]) * x + (-weights[0] / weights[2])
        
        plt.plot(x, y, c="r", label="Decision Boundary")
        plt.scatter(pos[:, 0], pos[:, 1], c="b", label=self.classes[1])
        plt.scatter(neg[:, 0], neg[:, 1], c="g", label=self.classes[0])
        plt.ylabel("Petal Length (cm)")
        plt.xlabel("Sepal Length (cm)")
        plt.grid()
        plt.legend()
        plt.show()
        
    '''
        plot the costs as a line graph
    '''
    def plot_cost(self, costs, file=None):
        plt.title("Error across Fitting")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.plot(np.arange(len(costs)), costs)
        if file is not None:
            plt.savefig(file)
        plt.show()

    
    #### LINEAR CLASSIFICATION FUNCTIONS ####
    
    '''
        closed form linear regression
        w = (xxT)**-1 * xy
    '''
    def closed_linear_clf(self):
        return np.dot(np.linalg.inv(np.dot(self.X_b.T, self.X_b)), np.dot(self.X_b.T, self.y))
    
    '''
        linear learning algorithm that processes each (xi, yi) one at a time
            and updates weight by:
            w(i + 1) = w(i) + ny(i)x(i) where () represents the subscript
        
        reduce prediction error: (w.T x(i) - y(i)) as each input data is processed
    '''
    def linear_clf(self, eta):
        w = np.zeros(self.X_b.shape[1])
        w_all = []
        cost = []

        for i, j in zip(self.X_b, self.y):
            y_pred = np.dot(w.T, i)
            error = y_pred - j
            cost.append(error**2)

            if not self.check_pred(y_pred, j):
                w_all.append(w)
                # the pdf description of updating weights
                #w += eta * j * i
                
                # a different implementation 
                w -= eta * np.dot(i.T, error)

        return w, cost

    '''
        logistic classifier using the standard classes 1 and 0 with this cost function:
            cost(h(x), y) = -log(h(x)) if y == 1 or -log(1 - h(x)) if y == 0
    '''
    def log_clf(self, eta, epochs=50):
        w = np.zeros(self.X_b.shape[1]) # + 1 for the bias term
        costs = []
                
        for n in range(epochs):
            for i in range(len(self.y)):
                # pick a random sample
                s = np.random.randint(0, len(self.y_2) - 1)      
                
                # calculate the probability of it being from class 1 w/ the current weights         
                sig = self.logistic(np.dot(self.X_b[s], w))
                
                # find the gradients of the cost function
                # gradient(cost) = XT(σ(Xθ) − y)
                grad = np.dot(self.X_b[s].T, sig - self.y_2[s])
                
                # loss = − y log(h(x)) − (1 − y)log(1 − h(x))
                # this is the same as the one described above, just in compact form
                error = (-self.y_2[s] * np.log(sig)) - ((1 - self.y_2[s]) * np.log(1 - sig))
                
                # adjust the weights
                w -= eta * grad
                
                costs.append(error**2)

        return w, costs
        
if __name__ == "__main__":
    clf = LinearRegression("iris.data")
    
    theta = clf.closed_linear_clf()
    print("Closed-Form Linear Regression")
    print("Accuracy Score: ", clf.accuracy(clf.predict(theta)))
    print("Final Weights: ", theta)
    clf.plot_samples(theta)
    print()
    
    theta, cost = clf.linear_clf(eta=0.1)
    print("Linear Learning Algorithm with learning rate of ", 0.1)
    print("Accuracy Score: ", clf.accuracy(clf.predict(theta)))
    print("Final Weights: ", theta)
    print()
    clf.plot_samples(theta)
    clf.plot_cost(cost)

    
    theta, cost = clf.linear_clf(eta=1.0)
    print("Linear Learning Algorithm with learning rate of ", 1.0)
    print("Accuracy Score: ", clf.accuracy(clf.predict(theta)))
    print("Final Weights: ", theta)
    print()
    clf.plot_samples(theta)
    clf.plot_cost(cost)

    theta, cost = clf.log_clf(0.1)
    print("Logistic Classifier with a learning rate of ", 0.1)
    print("Accuracy Score: ", clf.log_accuracy(clf.log_predict(theta, 0.50)))
    print("Final Weights: ", theta)
    print()
    clf.plot_samples(theta)
    clf.plot_cost(cost)