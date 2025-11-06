import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
      predictions = []

      for x in X:
          label = self._predict_one(x)
          predictions.append(label)

      return predictions


    def _predict_one(self, x):
        dist_list = []
        for index, train_x in enumerate(self.X_train):
          dist = self._distance(x, train_x)
          dist_list.append((dist, index))
        
        dist_list_sorted = sorted(dist_list)
        k_nearest = dist_list_sorted[:self.k] # we take the first k elements

        indices = [] #we take all the indices in k_nearest
        for (dist, idx) in k_nearest:
          indices.append(idx)
        
        neighbors_labels = []
        for i in indices:
          neighbors_labels.append(self.y_train[i])

        return Counter(neighbors_labels).most_common(1)[0][0]


    def _distance(self, a, b):
        diff = a - b
        dist = np.sqrt(np.sum(diff ** 2))
        return dist
