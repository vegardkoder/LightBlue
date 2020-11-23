#
# Author: Vegard Hansen Stenberg
# Date: 11.11.2020
# The activation functions for LightBlue
#

import numpy as  np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def ReLU(x):
    val = []
    for y in x:
        l  = []
        for z in y:
            l.append(max(0,z))
        val.append(l)
    return np.array(val)

if __name__ == '__main__':
    X_train = np.array([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1]])

    y_train = np.array([[0,1,1,0]]).T

    weights = 2 * np.random.random((3, 1)) - 1
    bias  = 1

    print(weights)

    outputs = ReLU(np.dot(X_train, weights)) + bias
    error = y_train - outputs
    print(outputs)
    print(Sigmoid_derivative(outputs))
    adjustments = error * Sigmoid_derivative(outputs)
    print(adjustments)

#ReLU(np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]]))
