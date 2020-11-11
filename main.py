#
# Author: Vegard Hansen Stenberg
# Date: 11.11.2020
# Goal: Create a basic NN to learn
#

import numpy as np
import activationFunctions

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Random starting synaptic weights:", synaptic_weights)

for iteration in range(100000):
    input_layer = training_inputs

    outputs = activationFunctions.ssigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * activationFunctions.sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Weigths after training:", synaptic_weights)

print("Outputs after training", outputs)
