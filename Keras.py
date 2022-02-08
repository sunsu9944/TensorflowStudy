import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

import numpy as np

print(tf.__version__)

x_data = np.array([1,2,3,4,5,6])
t_data = np.array([3,4,5,6,7,8])

model = Sequential()
model.add(Flatten(input_shape=(1,)))
model.add(Dense(1, activation='linear'))

model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')
model.summary()

hist = model.fit(x_data, t_data, epochs=1000)

result = model.predict(np.array([-3.1,3.0,3.5,15.0,20.1]))
print(result)