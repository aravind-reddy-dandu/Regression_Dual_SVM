import numpy as np
import pandas as pd
from pprint import pprint


def create_data(size):
    X_1_10 = np.random.normal(0, 1, (size, 11))
    sigma = np.sqrt(0.1)
    X_11 = np.asarray(
        [X_1_10[i][1] + X_1_10[i][2] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_12 = np.asarray(
        [X_1_10[i][3] + X_1_10[i][4] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_13 = np.asarray(
        [X_1_10[i][4] + X_1_10[i][5] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_14 = np.asarray([0.1 * X_1_10[i][7] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_15 = np.asarray(
        [2 * X_1_10[i][2] - 10 + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape((-1, 1))
    X_16_20 = np.random.normal(0, 1, (size, 5))

    return np.concatenate((X_1_10, X_11, X_12, X_13, X_14, X_15, X_16_20), axis=1)


def create_y(X, size):
    y = []
    sigma = np.sqrt(0.1)
    for i in range(size):
        temp = 10
        for j in range(1, 11):
            temp += (0.6 ** j) * X[i][j]
        temp += np.random.normal(loc=0, scale=sigma)
        y.append(temp)
    return np.asarray(y)


def create_dataset(m):
    X = create_data(m)
    y = create_y(X, m).reshape((m, 1))
    data = pd.DataFrame(np.append(X, y, axis=1), columns=["X" + str(i) for i in range(21)] + ['Y'])
    data['X0'] = 1
    return data


def mean_center_normalize(X_matrix, len_input):
    X_matrix = (X_matrix - np.mean(X_matrix, 0)) / np.std(X_matrix, 0)
    X_matrix = np.hstack((np.ones((len_input, 1)), X_matrix))
    return X_matrix


class LinearRegression:
    def __init__(self, X_matrix, Y_matrix, learning_rate=0.01, tot_iterations=1500):
        self.X = X_matrix
        self.Y = Y_matrix
        self.learning_rate = learning_rate
        self.iterations = tot_iterations

        self.num_samples = len(Y_matrix)
        self.num_features = X_matrix.shape[1]
        self.X = mean_center_normalize(self.X, self.num_samples)
        self.Y = Y_matrix[:, np.newaxis]
        self.weights = np.zeros((self.num_features + 1, 1))

    def fit_naive_reg(self):
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.Y)
        return self

    def fit_linear_reg_gradient(self):
        for _ in range(self.iterations):
            d = self.X.T @ (self.X @ self.weights - self.Y)
            self.weights = self.weights - (self.learning_rate / self.num_samples) * d
        return self

    def fit_ridge_reg(self, norm_const):
        n_samples, n_features = self.X.shape
        self.weights = np.dot(
            np.dot(np.linalg.inv(np.dot(self.X.T, self.X) + norm_const * np.identity(n_features)), self.X.T), self.Y)
        return self

    def fit_lasso(self, norm_const):
        n_samples, n_features = self.X.shape
        self.weights[0] = np.sum(self.Y - np.dot(self.X[:, 1:], self.weights[1:])) / n_samples
        for i in range(self.iterations):
            for j in range(1, n_features):
                copy_w = self.weights.copy()
                # copy_w[j] = 0.0
                residue = self.Y - np.dot(self.X, copy_w)
                first = np.dot(self.X[:, j], residue)
                second = norm_const / 2
                compare = (-first + second) / np.dot(self.X[:, j].T, self.X[:, j])
                compare_neg = (-first - second) / np.dot(self.X[:, j].T, self.X[:, j])
                if self.weights[j] > compare:
                    self.weights[j] = self.weights[j] - compare
                elif self.weights[j] < compare_neg:
                    self.weights[j] = self.weights[j] - compare_neg
                else:
                    self.weights[j] = 0
        return self

    def get_error(self, X_matrix=None, Y_matrix=None):
        if X_matrix is None:
            X_matrix = self.X
        else:
            X_matrix = mean_center_normalize(X_matrix, X_matrix.shape[0])

        if Y_matrix is None:
            Y_matrix = self.Y
        else:
            Y_matrix = Y_matrix[:, np.newaxis]

        y_pred = X_matrix @ self.weights
        score = 1 - (((Y_matrix - y_pred) ** 2).sum() / ((Y_matrix - Y_matrix.mean()) ** 2).sum())

        return score

    def predict(self, X):
        return mean_center_normalize(X, X.shape[0]) @ self.weights

    def get_weights(self):
        return self.weights[1:]


df_train = create_dataset(1000)
X = df_train.iloc[:, 1:-1]
y = df_train.iloc[:, -1]
linear_reg = LinearRegression(X, y).fit_lasso(50)
weights = linear_reg.get_weights()
df_test = create_dataset(1000)
# print(np.round(weights, 3))
pprint(weights.tolist())
print(linear_reg.get_error())
print(linear_reg.get_error(df_test.iloc[:, 1:-1], df_test.iloc[:, -1]))
