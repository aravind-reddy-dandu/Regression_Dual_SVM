import numpy as np
import pandas as pd
import Generate_data


# Class for implementing Barrier methods over Dual SVM
class Barrier_SVM:
    # Needs X data, Y data and few other non-mandatory values
    def __init__(self, X_matrix, Y_matrix, learning_rate=0.01, tot_iterations=1500):
        self.X = X_matrix
        self.Y = Y_matrix
        self.Y = Y_matrix[:, np.newaxis]
        self.learning_rate = learning_rate
        self.iterations = tot_iterations
        self.num_samples = len(Y_matrix)
        self.num_features = X_matrix.shape[1]
        self.weights = np.zeros((self.num_samples + 1, 1))

    # def fit_linear_reg_gradient(self):
    #     for _ in range(self.iterations):
    #         d = self.X.T @ (self.X @ self.weights - self.Y)
    #         self.weights = self.weights - (self.learning_rate / self.num_samples) * d
    #     return self

    def gradient_descent(self, epsilon, alpha, iterations):
        # Using gradient ascent to move towards maximum
        for _ in range(iterations):
            d = self.get_gradient_vector(x_bar=self.X[1:, :], y_bar=self.Y[1:, :], alpha_bar=alpha, epsilon=epsilon,
                                         x_1=self.X[1, :][:, np.newaxis].T, y_1=self.Y[1, :])
            # Formula for gradient ascent
            alpha = alpha + (self.learning_rate / self.num_samples) * d
        return alpha

    # Main Barrier method
    def barrier_method(self):
        # Initialize alpha with satisfying values
        # this code makes sure that sum of alpha_i+y_i is zero after initializing
        alpha = np.zeros(self.num_samples)
        unique, count = np.unique(self.Y, return_counts=True)
        for i, y in enumerate(self.Y):
            if y == 1:
                alpha[i] = count[np.where(unique == -1)] / count[np.where(unique == +1)]
            else:
                alpha[i] = 1
        # removing the first value
        alpha = alpha[1:]
        # using gradient ascent over a range of epsilon values
        for epsilon in range(10, 1000, 100):
            alpha = self.gradient_descent(epsilon, alpha, self.iterations)
        # Final alpha vector will the optimal value
        alpha_1 = -self.Y[0] * alpha @ self.Y[1:, :]
        # Finding alpha 1 from formula and adding it to optimal alpha
        final_alpha = np.concatenate((alpha_1, alpha), axis=1)
        return final_alpha

    # Function to get weights and bias from alpha values
    def get_weights_bias(self, alpha):
        # Using formula for w and b
        weight = (np.multiply(alpha, self.Y.T)) @ self.X
        bias = self.Y[0] - np.dot(weight, self.X[0])
        return weight, bias

    # Function to verify the fit and return error. Works on training data
    def verify_fit(self, weights, bias):
        y_pred = weights @ self.X.T
        y_pred = np.sign(y_pred)
        error = 0
        for i in range(len(self.Y)):
            if y_pred[0][i] != self.Y[i]:
                error += 1
        error /= len(self.Y)
        return error

    # Abandoned function
    def error_calculation(self, X, y, w):
        h_x = self.verify_fit(X, w)
        error = 0
        for i in range(len(y)):
            error += (y[i] - h_x[i]) ** 2
        error /= len(y)
        return error

    # def get_gradient_vector_old(self, X_from_2, Y_from_2):
    #     a1 = (- Y_from_2 * self.Y[0]) + (self.num_samples - 1)
    #     a2 = (-2 * (np.sum(self.weights @ (Y_from_2 * self.Y[0])))) * Y_from_2 * (np.dot(self.X[0].T, self.X[0]))
    #     a3 = (-2 * (np.sum(self.weights @ (Y_from_2 * self.Y[0])))) * self.Y[0] * (np.dot(Y_from_2.T, X_from_2)) * \
    #          self.X[0] + (-2 * Y_from_2 * self.Y[0]) * self.Y[0] * np.sum(
    #         np.dot(self.weights, np.dot(Y_from_2.T, np.dot(X_from_2.T, self.X[0]))))
    #     a4 = 0
    #     for i in range(2, self.num_samples + 1):
    #         for j in range(2, self.num_samples + 1):
    #             sum = self.weights[i] + self.weights[j] + (self.Y[i] * self.X[i] * self.Y[i] * self.X[i])

    # The most important method to calculate derivative for the complex function
    def get_gradient_vector(self, x_bar, y_bar, alpha_bar, epsilon, x_1, y_1):
        # Splitting each term and finding gradient for each term. Explained clearly in submission pdf
        g1 = -(y_1 * y_bar.T) + 1
        g2 = 2 * y_1 * np.dot(alpha_bar, y_bar) * (y_bar.T * y_1) * np.dot(x_1, x_1.T)
        f_g_3 = y_1 * np.dot(alpha_bar, y_bar)
        g_g_3 = np.sum(np.multiply(y_bar, np.dot(x_bar, x_bar.T)))
        g3 = -2 * y_1 * ((f_g_3 * (np.multiply(y_bar.T, x_1 @ x_bar.T))) + (g_g_3 * (y_bar.T * y_1)))
        g_4 = 2 * alpha_bar @ (np.multiply(y_bar @ y_bar.T, x_bar @ x_bar.T))
        g_5 = np.reciprocal(alpha_bar)
        k_g_6 = -y_1 * np.dot(alpha_bar, y_bar)
        g_6 = (-y_1 / k_g_6) * y_bar.T
        gradient = g1 - ((1 / 2) * (g2 + g3 + g_4)) + epsilon * (g_5 + g_6)
        return gradient


# Testing code snippet
errors = []
for i in range(10):
    # Using previous assignment code to generate positive and negative separated data
    data = Generate_data.gen_perceptron_data(e=0.5, m=100, k=5)
    # Generating X and Y from dataset
    X = np.asarray(data.iloc[:, : -1])
    Y = np.asarray(data.iloc[:, -1])
    # print(X)
    # print(Y)
    # calling barrier method using above data
    barrier = Barrier_SVM(X, Y)
    alphas = barrier.barrier_method()
    # print(alphas)
    # print(np.dot(alphas, Y))
    # Getting weights and bias from alphas
    weight, bias = barrier.get_weights_bias(alphas)
    # print(weight, bias)
    # Getting error and appending to global list
    errors.append(barrier.verify_fit(weight, bias))
    print(errors)
print(errors)
