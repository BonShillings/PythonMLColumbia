# Author Sean Billings

# Input: X_train.csv y_train.csv X_test.csv
#
# Output:   probs_test.csv     column vector of posterior probabilities for X_test
#
# Assumptions: y in (1,10) , y~iid~DISCRETE(Pr)
#              Pr(X|y) ~ Normal(mean_y,sigma_y)

import sys
import math
import numpy as np

arg_list = sys.argv
X_train_csv = open(arg_list[1])
y_train_csv = open(arg_list[2])
X_test_csv = open(arg_list[3])

X = np.genfromtxt(X_train_csv, delimiter=",")
y = np.genfromtxt(y_train_csv, delimiter=",")

len_X, dim_X = X.shape[0], X.shape[1]

# compute prior on X,Y
count_y = np.zeros(10)
for i in range(0, y.shape[0]):
    count_y[y[i]] += 1

pr_y = np.divide(count_y, len_X)  # len_X = len_y := number of observations
print(pr_y)

# compute mean_k
mean_k = np.zeros((10, dim_X))  # row vector
for i in range(0, len_X):
    mean_k[y[i]] = np.add(mean_k[y[i]], X[i])

mean_k = np.divide(mean_k, count_y)

# compute Sigma_k
sigma_k = np.zeros((10, dim_X, dim_X))  # 10 sigma tensor
for i in range(0, len_X):
    sigma_k[y[i]] = np.add(sigma_k[y[i]], np.outer(np.transpose(X[i] - mean_k[y[i]]), (X[i] - mean_k[y[i]])))
    # X[i]-mean_y[y[i]] is a row vector so we must transpose first for outer product

for i in range(0, count_y.shape[0]):
    sigma_k[i] = np.divide(sigma_k[i], count_y[i])

    # X[i]-mean_y[y[i]] is a row vector so we must transpose first for outer product

# evaluate classifier on X_test

X_test = np.genfromtxt(X_test_csv, delimiter=",")
len_X_test, dim_X_test = X_test.shape[0], X_test.shape[1]

sigma_inverse_k = np.zeros((10, dim_X, dim_X))
for i in range(0, sigma_k.shape[0]):
    sigma_inverse_k[i] = np.linalg.inv(sigma_k[i])

probs_test_out = open("probs_test.csv", 'w')

for i in range(0, len_X_test):
    for k in range(0, sigma_k.shape[0]):
        # evalute prior,gaussian to calculate posterior
        norm_sigma_k = np.linalg.norm(sigma_k[k]) ** (-1 / 2)
        gaussian_arg = np.dot(np.transpose(X_test[i] - mean_k[k]), np.dot(sigma_inverse_k[k], (X_test[i] - mean_k[k])))
        gaussian_k = np.exp(-1 / 2 * gaussian_arg)
        p_k_x = pr_y[k] * norm_sigma_k * gaussian_k

        probs_test_out.write(str(p_k_x))
        if (k < sigma_k.shape[0] - 1):
            probs_test_out.write(",")

    probs_test_out.write(str("\n"))

probs_test_out.close()