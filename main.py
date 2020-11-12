#
# Author: Vegard Hansen Stenberg
# Date: 11.11.2020
# Goal: Create a basic NN to learn
#

import numpy as np
import activationFunctions
import timeit

# ---------- training ----------

X_train = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

y_train = np.array([[0,1,1,0]]).T

weights = 2 * np.random.random((3, 1)) - 1

for i in range(100000):
    outputs = activationFunctions.Sigmoid(np.dot(X_train, weights))
    error = y_train - outputs
    adjustments = error * activationFunctions.Sigmoid_derivative(outputs)
    weights += np.dot(X_train.T, adjustments)

results = np.around(outputs).astype(int)

print("Weigths after training: \n", weights)
print("Outputs after training: \n", outputs)
print("Results after training: \n", results)

# ---------- custom inputs ----------

user_input = np.array([1,1,0])
user_output = activationFunctions.Sigmoid(np.dot(user_input, weights))
user_result = np.around(user_output).astype(int)

print("User outputs: \n", user_output)
print("User result: \n", user_result)

# ---------- timing the execution ----------

print(f"Execution time: {timeit.timeit()}")
