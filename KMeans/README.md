# K-Means Clustering From Scratch

This folder contains a complete implementation of the **K-Means Clustering** algorithm using only Python and NumPy. This is an unsupervised learning model designed to partition data into $K$ distinct clusters.

## 📌 Contents

* **KMeans_from_scratch.ipynb** — Notebook with data generation, model training, and cluster visualization.
* **KMeans.py** — Standalone Python module containing the `KMeans` class.
* **README.md** — Project documentation.

## 🚀 Features

* **Centroid Initialization**: Randomly selects $K$ starting points from the dataset.
* **Iterative Optimization**: Implements the classic Expectation-Maximization (EM) logic.
* **Convergence Detection**: Automatically stops training when centroids no longer move.
* **Flexible Predictions**: Assigns new, unseen data points to the nearest existing cluster.

---

## 📊 Mathematical Foundation

The algorithm works by minimizing the distance between points and their assigned cluster center.

### 1. Distance Metric (Euclidean Distance)
To determine the closest cluster for each point, we calculate the straight-line distance:
$$d(x, c) = \sqrt{\sum_{i=1}^{n} (x_i - c_i)^2}$$

### 2. Objective Function (Inertia)
The goal is to minimize the **Within-Cluster Sum of Squares (WCSS)**, which measures how tightly the clusters are packed:
$$WCSS = \sum_{j=1}^{K} \sum_{i \in C_j} ||x_i - \mu_j||^2$$
*Where $\mu_j$ is the mean (centroid) of all points in cluster $C_j$.*


## 🛠️ How to Use

```python
from KMeans import KMeans
import numpy as np

# Initialize and fit
model = KMeans(k=4, max_iters=100)
model.fit(X)

# Get cluster assignments
labels = model.predict(X)
