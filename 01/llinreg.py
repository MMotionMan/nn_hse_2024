import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.01, num_iterations=500):
        self.lr = lr
        self.num_iterations = num_iterations
        self.W = None
        self.b = None

    def fit(self, X, y):
        num_targets = y.shape[1]
        num_features = X.shape[1]
        self.W = np.random.random((num_targets, num_features))
        self.b = np.random.random((num_targets, 1))

        self.cost_f = []

        for i in range(self.num_iterations):
            pred = X.dot(self.W.T) + self.b.T

            grad_W = -(2 / X.shape[0]) * (y - pred).T.dot(X)
            grad_b = -(2 / X.shape[0]) * np.sum(y - pred, axis=0, keepdims=True).T

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            self.cost_f.append(np.mean((y - pred) ** 2))

    def predict(self, X):
        return X.dot(self.W.T) + self.b.T


num_examples = 100
num_features = 3
num_targets = 2

X = np.random.randn(num_examples, num_features)
W_true = np.random.randn(num_targets, num_features)
b_true = np.random.randn(num_targets, 1)
y_true = X.dot(W_true.T) + b_true.T
noise = 0.3 * np.random.randn(num_examples, num_targets)
y = y_true + noise

linreg = LinearRegression()

linreg.fit(X, y_true)
# print(linreg.cost_f)
#
# plt.plot([i for i in range(0, len(linreg.cost_f))], linreg.cost_f)
# plt.show()
