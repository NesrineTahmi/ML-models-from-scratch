import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        #we flatten y to ensure it's a 1D vector and avoid broadcasting errors
        y_true = y.flatten()

        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            #save cost
            error = y_predicted - y_true
            cost = (1 / (2 * n_samples)) * np.sum(error**2)
            self.cost_history.append(cost)

            #Gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            #Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias