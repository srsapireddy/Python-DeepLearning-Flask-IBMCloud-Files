# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:04:25 2020

@author: prads
"""
import numpy as np
import pandas as pd
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t")

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data = []
for i in range(0,1000):
    review = dataset['Review'][i]
    review  = re.sub('[^a-zA-z]',' ',review)
    review = review.lower()
    review = review.split()
    review  = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()

y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.2,random_state = 0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 1565,init = 'uniform',activation = 'relu'))
model.add(Dense(units = 3000,init = 'uniform',activation = 'relu'))
model.add(Dense(units = 1,init = 'uniform',activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = "binary_crossentropy",metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 10,batch_size = 32)

y_pred = model.predict(x_test)
y_pred = y_pred>0.5
text = "loved"
y = model.predict(cv.transform([text]))

text = "it was awesome i cannot take it ........."

text  = re.sub('[^a-zA-z]',' ',text)
text = text.lower()
text = text.split()
text  = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)

y = model.predict(cv.transform([text]))

y>0.5












