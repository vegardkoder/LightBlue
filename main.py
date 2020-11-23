#
# Author: Vegard Hansen Stenberg
# Date: 11.11.2020
# Goal: Create a basic NN to learn
#

import numpy as np

import activationFunctions
import models

X_train = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

y_train = np.array([[0,1,1,0]]).T

model = models.Sigmoid()

model.train(X=X_train, y=y_train, iterations=1000)

user_input = np.array([0,1,0])
prediction = model.predict(user_input)

print(prediction)
