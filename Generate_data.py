import numpy as np
import itertools
from pprint import pprint
import pandas as pd


# Function to generate perceptron data. Takes epsilon, size of data and dimension as inputs
def gen_perceptron_data(e, m, k):
    dataset = {}
    # Initializing dataset storage with header
    for x in range(1, k + 1):
        dataset['x' + str(x)] = []
    dataset['y'] = []
    # Looping until dataset reaches required size which is m
    while len(dataset['x1']) < m:
        # Drawing k samples out of normal distribution using mean 0 and variance 1
        z = np.random.normal(0, 1, k)
        # Taking euclidean norm of Z
        z_norm = np.linalg.norm(z, 2)
        # Dividing Z vector with norm of Z to get X_from_2
        x = z / z_norm
        y = None
        # checking if absolute value is greater than epsilon. Continuing otherwise
        if abs(x[len(x) - 1]) > e:
            # Assigning positive if xk is positive(will obviously be greater than e). Negative else
            if x[len(x) - 1] > e:
                y = 1
            else:
                y = -1
        else:
            continue
        # Including X_from_2 vector in dataset
        for index, xk in enumerate(x):
            dataset['x' + str(index + 1)].append(xk)
        # Appending Y_from_2 values to last column
        dataset['y'].append(y)
    # Returning a pandas dataframe
    dataset = pd.DataFrame.from_dict(dataset)
    return dataset


# df = gen_perceptron_data(e=0.2, m=100, k=5)
# pprint(df)
# pn = Perceptron()
# num_steps = pn.fit_perceptron(df)
# error = pn.eveluate_perceptron(df)
# print(error)