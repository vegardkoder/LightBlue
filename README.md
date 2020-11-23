# LightBlue

Have you ever felt the need for using a
super simple ML libaray? Sure, me too. LightBlue helps
with easy predictions and the uses are currently very
limited.

## Avalible models:
Sigmoid
AverageRegression

## Example for sigmoid
```python
import numpy as np

import activationFunctions
import models

X_train = np.array([[0,0,1],
                    [1,1,1],
                    [1,0,1],
                    [0,1,1]])

y_train = np.array([[0,1,1,0]]).T

model = models.Sigmoid()

model.train(X_train, y_train, 10000)

user_input = np.array([0,1,0])
prediction = model.predict(user_input)

print(prediction)
