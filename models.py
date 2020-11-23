import numpy as np
import activationFunctions

class Sigmoid():
    def __init__(self):
        self.weights = 2 * np.random.random((3, 1)) - 1
        print("Initialized Sigmoid model.")

    def train(self, X, y, iterations):
        for i in range(iterations):
            outputs = activationFunctions.Sigmoid(np.dot(X, self.weights))
            error = y - outputs
            adjustments = error * activationFunctions.Sigmoid_derivative(outputs)
            self.weights += np.dot(X.T, adjustments)
        print("Training complete")

    def predict(self, X):
        return activationFunctions.Sigmoid(np.dot(X, self.weights))

class LinearRegression():
    def __init__(self):
        print("Initialized LinearRegression model.")

class ReLU():
    def __init__(self):
        print("Initialized ReLU model.")
