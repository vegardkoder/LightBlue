import numpy as  np

def Sigmoid(x):
    print(1 / (1 + np.exp(-x)))
    return 1 / (1 + np.exp(-x))

def Sigmoid_derivative(x):
    return x * (1 - x)

def ReLU(x):
    val = []
    for y in x:
        l  = []
        for z in y:
            l.append(max(0,z))
        val.append(l)
    return np.array(val)

#ReLU(np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]]))
