import pandas as pd
import numpy as np

data = pd.read_csv('Data/gpascore.csv')
print(data)

#data 전처리

data.isnull().sum()
data = data.dropna()

y_result = data['admit'].values

x_data = []

for i, rows in data.iterrows():
   x_data.append([rows['gre'],rows['gpa'],rows['rank']])


import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation = 'tanh'),
    tf.keras.layers.Dense(657, activation = 'tanh'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x_data),np.array(y_result),epochs = 1000)

# 실험데이터 [[380,3.21,3] [ 660,3.67,3]]
# 결과데이터 [[0],[1],[0],[1]]

#예측
predict = model.predict( [ [750,3.70,3],[400,2.2,1] ] )

print(predict)


