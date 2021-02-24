# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:23:25 2020

@author: prads
"""
#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the Dataaset 
dataset = pd.read_csv("Google_Stock_Price_Train.csv")

train_set = dataset.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
Scaled_training = sc.fit_transform(train_set)

x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(Scaled_training[i-60:i,0])
    y_train.append(Scaled_training[i,0])



x_train,y_train = np.array(x_train),np.array(y_train)

type(x_train)

print(x_train.ndim)



x_train = np.reshape(x_train ,(1198,60,1))

x_train.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = Sequential()

model .add(LSTM(units = 50, input_shape  =  (x_train.shape[1],1),return_sequences = True))
model.add(Dropout(0.2))
model .add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model .add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model .add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units= 1))

model.compile(optimizer = 'adam', loss = 'mse')

model.fit(x_train,y_train, epochs = 1 , batch_size = 32)

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
dataset .shape
dataset_test.shape 
dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis = 0)
dataset_total.shape
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

#len(dataset_total) = 1278
#len(dataset_test) = 20
#1158 - 60 = 1198

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test.shape
x_test = np.reshape(x_test,(20,60,1))

preicted_stock_price = model.predict(x_test)
sc.inverse_transform(preicted_stock_price) 
plt.plot(real_stock_price , color = 'red' , label = 'real google stok price')
plt.plot(sc.inverse_transform(preicted_stock_price), color = 'blue' , label = 'pred google stok price')

plt.title("stock")
plt.xlabel('timr')
plt.ylabel('gst')
plt.legend()
plt.show()

#


