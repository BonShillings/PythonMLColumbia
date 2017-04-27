#Author Sean Billings

# Input: lambda sigma2 X_train.csv y_train.csv X_test.csv
#
# Output:   wRR_out     column vector of wRR
#           active_out  list of 10 vectors to active learn


import sys
import math
import numpy as np


def parse_matrix_from_csv(file_in):
    dim = 0
    first = next(file_in)
    X_np = np.array([float(b) for b in first.split(',')])
    dim = X_np.size
    for line in file_in:
        dim += 1
        np.append(X_np, np.array([float(b) for b in line.split(",")]))

    return np.matrix(X_np), dim



arg_list = sys.argv
lambda_in = arg_list[1]
sigma2 = arg_list[2]
X_train_csv = open(arg_list[3])
y_train_csv = open(arg_list[4])
X_test_csv = open(arg_list[5])

wRR_out = open("wRR_" + lambda_in + ".csv", 'w')
active_out = open("active_" + lambda_in +  "_" + sigma2 + ".csv", 'w')


# read X, y

X = np.genfromtxt(X_train_csv,delimiter=",")
y = np.genfromtxt(y_train_csv,delimiter=",")
dim_X = X.shape[1]
#X, dim_x = parse_matrix_from_csv(X_train_csv)
#y, dim_y = parse_matrix_from_csv(y_train_csv)

# compute w = (X^TX + lamba*I)^-1 * X^T * Y
X_T = np.transpose(X)
lambda_I = float(lambda_in) * np.eye(dim_X, dtype=float)
X_gramian = np.dot(X_T, X)
inner = X_gramian + lambda_I

core = np.linalg.inv(inner)

w = np.dot(np.dot(core,X_T), y)

for element in w:
    wRR_out.write(str(element) + '\n')


# varW = (X^TX * 1/sigma2 + lamba*I)^-1
varW_inner = X_gramian * ( 1/float(sigma2) )

varW_inner += lambda_I
varW = np.linalg.inv(varW_inner)


first_x_test = next(X_test_csv)
x_test_1 = np.array([float(b) for b in first_x_test.split(',')])
i = 1
x1_score_m = np.dot(np.dot(np.transpose(x_test_1),varW), x_test_1)
x1_score = x1_score_m + float(sigma2)
xi_scores = [[1, x1_score]]
for line in X_test_csv:
    i += 1
    x_test_i = np.array([float(b) for b in line.split(",")])
    xi_score_m = np.dot(np.dot(np.transpose(x_test_i),varW), x_test_i)
    xi_score = xi_score_m + float(sigma2)
    xi_scores.append([i,xi_score])


xi_scores_sorted = sorted(xi_scores, key=lambda x: x[1], reverse=True)

for j in range(1,11):
    print(str(xi_scores_sorted[j][1]))
    active_out.write(str(xi_scores_sorted[j][0]) +',')