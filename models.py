import numpy as np
import activationFunctions

class Sigmoid():
    def __init__(self):
        self.weights = 2 * np.random.random((3, 1)) - 1
        print("Initialized Sigmoid model.")

    def train(self, X, y, iterations):
        for i in range(iterations):
            outputs = activationFunctions.sigmoid(np.dot(X, self.weights))
            error = y - outputs
            adjustments = error * activationFunctions.sigmoid_derivative(outputs)
            self.weights += np.dot(X.T, adjustments)
        print("Training complete")

    def predict(self, X):
        return activationFunctions.Sigmoid(np.dot(X, self.weights))

class AverageRegression():
    def __init__(self):
        print("Initialized LinearRegression model.")

    def train(self, X):
        s = 0
        for x in X:
            s += x[1] / x[0]
        self.weight = s / len(X)
        print("Training complete.")

    def pred(self, X):
        return X * self.weight;

class ReLU():
    def __init__(self):
        print("Initialized ReLU model.")
