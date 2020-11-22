import numpy as np
import activationFunctions

class Sigmoid():

    def __init__(self):
        print("Initialized Sigmoid model.")


    def train(self, X, y, iterations):
        weights = 2 * np.random.random((3, 1)) - 1
        for i in range(iterations):
            outputs = activationFunctions.Sigmoid(np.dot(X, weights))
            error = y - outputs
            adjustments = error * activationFunctions.Sigmoid_derivative(outputs)
            weights += np.dot(X, adjustments)


model = Sigmoid()
