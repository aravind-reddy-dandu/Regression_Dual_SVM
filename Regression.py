import numpy as np
import pandas as pd
from pprint import pprint


# Function to create X data as defined
def create_x(size):
    # Values from 1 through 10 are normal random values
    X_1to10 = np.random.normal(0, 1, (size, 10))
    # Defining standard deviation
    sigma = np.sqrt(0.1)
    # Using given formulae for x_11, x12_....
    X_11 = np.asarray(
        [X_1to10[i][1] + X_1to10[i][2] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_12 = np.asarray(
        [X_1to10[i][3] + X_1to10[i][4] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_13 = np.asarray(
        [X_1to10[i][4] + X_1to10[i][5] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_14 = np.asarray([0.1 * X_1to10[i][7] + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape(
        (-1, 1))
    X_15 = np.asarray(
        [2 * X_1to10[i][2] - 10 + np.random.normal(loc=0, scale=sigma) for i in range(size)]).reshape((-1, 1))
    # x_16 through x_20 are again normal random values
    X_16to20 = np.random.normal(0, 1, (size, 5))
    # Concatenating all values to create one single array of X values
    return np.concatenate((X_1to10, X_11, X_12, X_13, X_14, X_15, X_16to20), axis=1)


# Function to generate true weights for Xs using formula
def generate_true_weights():
    w_actual = []
    # Using formula to generate weights
    for i in range(1, 21):
        if i <= 10:
            w_actual.append(0.6 ** i)
        else:
            w_actual.append(0)
    return w_actual


# Function to generate Y values given X matrix
def create_y(X, size):
    y = []
    sigma = np.sqrt(0.1)
    # Creating y value for each row
    for i in range(size):
        randomness = 10
        for j in range(1, 11):
            randomness += (0.6 ** j) * X[i][j - 1]
        randomness += np.random.normal(loc=0, scale=sigma)
        y.append(randomness)
    # Returns y values in a np array
    return np.asarray(y)


# Function to merge X and Y data and create a pandas dataframe
def merge_x_y_data(m):
    X = create_x(m)
    y = create_y(X, m).reshape((m, 1))
    data = pd.DataFrame(np.append(X, y, axis=1), columns=["X" + str(i + 1) for i in range(20)] + ['Y'])
    return data


# This is used to get the X data to the center and add bias. Mean centering is recommended to give better results and
# less worry about bias
def mean_center_normalize(X_matrix, len_input):
    # Squashing whole data between 0 and 1
    X_matrix = (X_matrix - np.mean(X_matrix, 0)) / np.std(X_matrix, 0)
    # Adding 1s as the first column to represent bias
    X_matrix = np.hstack((np.ones((len_input, 1)), X_matrix))
    return X_matrix


# Class to perform different types of regression
class LinearRegression:
    # Init method. Needs X data, Y data and other optional parameters
    def __init__(self, X_matrix, Y_matrix, learning_rate=0.01, tot_iterations=1500):
        self.X = X_matrix
        self.Y = Y_matrix
        self.learning_rate = learning_rate
        self.iterations = tot_iterations
        self.num_samples = len(Y_matrix)
        self.num_features = X_matrix.shape[1]
        # Mean centering given data
        self.X = mean_center_normalize(self.X, self.num_samples)
        self.Y = Y_matrix[:, np.newaxis]
        # Initializing weights to a zero vector
        self.weights = np.zeros((self.num_features + 1, 1))

    # Simple formula to fit data to naive regression. No gradient descent used. Assuming XTX is invertible
    def fit_naive_reg(self):
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), self.X.T), self.Y)
        return self

    # Using gradient descent in linear regression method
    def fit_linear_reg_gradient(self):
        for _ in range(self.iterations):
            # Finding gradient and moving towards minimization
            d = self.X.T @ (self.X @ self.weights - self.Y)
            self.weights = self.weights - (self.learning_rate / self.num_samples) * d
        return self

    # Using simple formula for Ridge regression
    def fit_ridge_reg(self, norm_const):
        n_samples, n_features = self.X.shape
        self.weights = np.dot(
            np.dot(np.linalg.inv(np.dot(self.X.T, self.X) + norm_const * np.identity(n_features)), self.X.T), self.Y)
        return self

    # Using formulae given in class notes for Lasso regression
    def fit_lasso(self, norm_const):
        n_samples, n_features = self.X.shape
        # calculating bias using formula. No iterations needed as this is independent
        self.weights[0] = np.sum(self.Y - np.dot(self.X[:, 1:], self.weights[1:])) / n_samples
        for i in range(self.iterations):
            for j in range(1, n_features):
                # Maintaining a copy of weights. Not actually needed
                copy_w = self.weights.copy()
                residue = self.Y - np.dot(self.X, copy_w)
                # Computing first value in numerator of formula
                first = np.dot(self.X[:, j], residue)
                # Computing second value in numerator of formula
                second = norm_const / 2
                # These are used to check for conditions
                compare = (-first + second) / np.dot(self.X[:, j].T, self.X[:, j])
                compare_neg = (-first - second) / np.dot(self.X[:, j].T, self.X[:, j])
                # Updating weights based on conditions
                if self.weights[j] > compare:
                    self.weights[j] = self.weights[j] - compare
                elif self.weights[j] < compare_neg:
                    self.weights[j] = self.weights[j] - compare_neg
                else:
                    self.weights[j] = 0
        return self

    # By default returns training error. Takes X and Y data as input
    def get_error(self, X_matrix=None, Y_matrix=None):
        # If no data given, calculating training error
        # mean-centering
        if X_matrix is None:
            X_matrix = self.X
        else:
            # Mean centering any input data as we've found weights after this
            X_matrix = mean_center_normalize(X_matrix, X_matrix.shape[0])

        if Y_matrix is None:
            Y_matrix = self.Y
        else:
            Y_matrix = Y_matrix[:, np.newaxis]

        # Using the formula to find Y values. Bias is eliminated here(Mean centering)
        y_pred = X_matrix @ self.weights
        # Returning scaled score for better understanding. Error is squashed between 0 and 1
        # Example: Error of 0.1 is less. Error of 0.9 is terrible
        score = (((Y_matrix - y_pred) ** 2).sum() / ((Y_matrix - Y_matrix.mean()) ** 2).sum())
        return score

    # Simple predict function.
    def predict(self, X):
        return mean_center_normalize(X, X.shape[0]) @ self.weights

    # Method exposed to return weights after training
    def get_weights(self):
        return self.weights[1:]


# df_train = merge_x_y_data(1000)
# X = df_train.iloc[:, 1:-1]
# y = df_train.iloc[:, -1]
# linear_reg = LinearRegression(X, y).fit_linear_reg_gradient()
# weights = linear_reg.get_weights()
# df_test = merge_x_y_data(1000)
# # print(np.round(weights, 3))
# pprint(weights.tolist())
# print(linear_reg.get_error())
# print(linear_reg.get_error(df_test.iloc[:, 1:-1], df_test.iloc[:, -1]))

# First question
def test_naive_reg():
    # Generating data
    df_train = merge_x_y_data(10000)
    # Getting X and Y from data. Last col is Y. Remaining is X
    X = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1]
    # linear_reg = LinearRegression(X, y).fit_lasso(norm_const=50)
    # Calling naive regression
    linear_reg = LinearRegression(X, y).fit_naive_reg()
    weights = linear_reg.get_weights()
    print(weights)
    print(generate_true_weights())
    # pprint(dict(zip(list(X.columns), linear_reg.get_weights().tolist())))
    # Creating a df to plot computed and true weights
    weight_df = pd.DataFrame(list(zip(list(X.columns), weights, generate_true_weights())),
                             columns=['index', 'Predicted', 'Actual'])
    weight_df = pd.concat(
        (weight_df['index'], pd.DataFrame(weight_df['Predicted'].tolist(), columns=['Predicted']), weight_df['Actual']),
        axis=1)
    # Storing data in a csv. Will be used for plotting
    weight_df.to_csv('D:\\Study\\ML\\Computations_Assignment\\CSV_Files\\first.csv', index=False)
    pprint({list(X.columns)[i]: linear_reg.get_weights().tolist()[i] for i in range(len(list(X.columns)))})
    print('Training scaled error of Naive Regression is ', linear_reg.get_error())
    # Generating huge data to get True error by testing dataset
    df_test = merge_x_y_data(10000)
    print('True scaled error of Naive Regression is ', linear_reg.get_error(df_test.iloc[:, :-1], df_test.iloc[:, -1]))


# Function to test ridge regression. Second question
def test_ridge_reg():
    # Getting data
    df_train = merge_x_y_data(1000)
    X = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1]
    store = {}
    # Iterating over different values of lambda
    for norm_const in range(1, 20, 1):
        norm_const = norm_const
        linear_reg = LinearRegression(X, y).fit_ridge_reg(norm_const)
        weights = linear_reg.get_weights()
        # print(weights)
        # print(generate_true_weights())
        # pprint(dict(zip(list(X.columns), linear_reg.get_weights().tolist())))
        weight_df = pd.DataFrame(list(zip(list(X.columns), weights, generate_true_weights())),
                                 columns=['index', 'Predicted', 'Actual'])
        weight_df = pd.concat(
            (weight_df['index'], pd.DataFrame(weight_df['Predicted'].tolist(), columns=['Predicted']),
             weight_df['Actual']),
            axis=1)
        print(weight_df)
        print('Training scaled error of Ridge Regression is ', linear_reg.get_error())
        df_test = merge_x_y_data(100000)
        true_error = linear_reg.get_error(df_test.iloc[:, :-1], df_test.iloc[:, -1])
        print('True scaled error of Ridge Regression is ', true_error)
        store[norm_const] = true_error
    # print(store)
    for val in store:
        print(val, ',', store[val])


# testing Lasso regression. Third question
def test_lasso():
    df_train = merge_x_y_data(1000)
    X = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1]
    store = {}
    error_store = {}
    # Iterating over a range of lambda.
    for norm_const in range(100, 1000, 100):
        linear_reg = LinearRegression(X, y).fit_lasso(norm_const)
        weights = linear_reg.get_weights()
        zero_count = 0
        for weight in weights:
            if weight[0] == 0:
                zero_count += 1
        df_test = merge_x_y_data(10000)
        true_error = linear_reg.get_error(df_test.iloc[:, :-1], df_test.iloc[:, -1])
        print('True scaled error of Lasso Regression is ', true_error)
        store[norm_const] = zero_count
        error_store[norm_const] = true_error
        print(norm_const, zero_count)
    # Printing performance over a range of lambda values
    for val in error_store:
        print(val, ',', error_store[val])


# Mixing Lasso and Ridge. Using Lasso for feature selection
def mix_ridge_lasso():
    df_train = merge_x_y_data(1000)
    X = df_train.iloc[:, :-1]
    y = df_train.iloc[:, -1]
    lasso_reg = LinearRegression(X, y).fit_lasso(60)
    weights = lasso_reg.get_weights()
    weight_df = pd.DataFrame(list(zip(list(X.columns), weights, generate_true_weights())),
                             columns=['index', 'Predicted', 'Actual'])
    weight_df = pd.concat(
        (weight_df['index'], pd.DataFrame(weight_df['Predicted'].tolist(), columns=['Predicted'])),
        axis=1)
    weight_df = weight_df.where(weight_df['Predicted'] != 0).dropna()
    df_test = merge_x_y_data(10000)
    true_error = lasso_reg.get_error(df_test.iloc[:, :-1], df_test.iloc[:, -1])
    print('True scaled error of Lasso Regression is ', true_error)
    print('Important features according to Lasso are ', weight_df['index'].tolist())
    ridge_reg = LinearRegression(X[weight_df['index']], y).fit_ridge_reg(5)
    true_error = ridge_reg.get_error(df_test[weight_df['index']], df_test.iloc[:, -1])
    print('True scaled error of Ridge-Lasso Regression is ', true_error)


mix_ridge_lasso()
