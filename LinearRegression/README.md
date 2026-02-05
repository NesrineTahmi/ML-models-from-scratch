# Linear Regression From Scratch

This folder contains a full implementation of the **Linear Regression** algorithm using only Python and NumPy (no scikit-learn is used for the model logic).

## 📌 Contents

* **LinearRegression_from_scratch.ipynb** — Complete implementation, testing on synthetic data, and visualizations.
* **Linear_Regression.py** — Standalone Python module containing the `LinearRegression` class with Gradient Descent.
* **README.md** — This documentation.

## 🚀 Features

* **Gradient Descent Optimization**: Manual implementation of the update rules for weights and bias.
* **Cost Tracking**: Stores the Mean Squared Error (MSE) at each iteration to visualize convergence.
* **Custom Metrics**: Full implementation of evaluation scores from scratch.
* **Broadcasting Safety**: Robust handling of NumPy arrays to prevent dimension errors.


## 📊 Mathematical Metrics

To evaluate our model, we use the following metrics implemented in the code:

### 1. Mean Squared Error (MSE)
The average of the squared differences between predicted and actual values.
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 2. Root Mean Squared Error (RMSE)
The square root of the MSE, providing an error metric in the same units as the target variable.
$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 3. R² Score (Coefficient of Determination)
Indicates how much of the variance in the target variable is explained by the model compared to a simple mean.
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
*Where:*
* **$SS_{res}$** (Residual Sum of Squares): $\sum (y_i - \hat{y}_i)^2$
* **$SS_{tot}$** (Total Sum of Squares): $\sum (y_i - \bar{y})^2$

---

## 📈 Performance
On a standard synthetic regression dataset with noise, this model typically achieves:
* **R² Score:** ~0.77+
* **MSE:** ~417 (depending on the noise level)
