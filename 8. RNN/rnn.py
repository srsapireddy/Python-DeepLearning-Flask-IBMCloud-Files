# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:10:35 2020

@author: Rahul Sapireddy
"""

# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

train_set = dataset_train.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
scaled_training = sc.fit_transform(train_set)

x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(scaled_training[i-60:i, 0])
    y_train.append(scaled_training[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train.shape
x_train.ndim
x_train = np.reshape(x_train, (1198, 60, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()
model.add((LSTM(units = 60, return_sequences = True, input_shape = (60,1))))
model.add(Dropout(0.2))
model.add(LSTM(units = 60, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 60, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 60))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = "adam", loss = "mse", metrics = ["mse"])
model.fit(x_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")

real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :]
inputs.shape

inputs = np.array(inputs)
inputs = inputs.reshape(-1, 1)
inputs.shape
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i - 60 : i, 0])

x_test = np.array(x_test)
x_test.shape
x_test = np.reshape(x_test, (20, 60, 1))
x_test.shape

predicted_stock_price = model.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = "red", label = "real price")
plt.plot(predicted_stock_price, color = "blue", label = "pred price")
plt.legend()
plt.show()














    


