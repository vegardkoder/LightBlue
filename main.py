#
# Author: Vegard Hansen Stenberg
# Date: 11.11.2020
# Goal: Create a basic NN to learn
#

import numpy as np
import activationFunctions

X_train = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

y_train = np.array([[0,1,1,0]]).T

weights = 2 * np.random.random((3, 1)) - 1

for iteration in range(100000):
    outputs = activationFunctions.Sigmoid(np.dot(X_train, weights))
    error = y_train - outputs
    adjustments = error * activationFunctions.Sigmoid_derivative(outputs)
    weights += np.dot(X_train.T, adjustments)

results = np.around(outputs).astype(int)

print("Weigths after training: \n", weights)
print("Outputs after training: \n", outputs)
print("Results after training: \n", results)

#user_input = np.array(input("Give me an array"))
#print(user_input[0])
