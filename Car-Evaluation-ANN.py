#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import theano
import tensorflow
from keras import *

#import the data into dataset dataframe from the csv file
column_headers = ['buying','maint','doors','persons','lug_boot','safety','class']
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',names = column_headers,delimiter=",")

#import library to convert categorical values into encoded form
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#convert the dataset dataframe into two numpy arrays of feature variable and target variable
X = dataset.iloc[:,0:6].values
Y = dataset.iloc[:,6]

labelencoder = LabelEncoder()
X[:,0] =labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,4] = labelencoder.fit_transform(X[:,4])
X[:,5] = labelencoder.fit_transform(X[:,5])

for i in range(len(X)):
    if X[i][2] == '5more':
        X[i][2] = 5
for i in range(len(X)):
    if X[i][3] == 'more':
        X[i][3] = 5

d1 = pd.DataFrame(X,columns= ['buying','maint','doors','persons','lug_boot','safety'])

d2 = pd.get_dummies(d1,columns= ['buying','maint','lug_boot','safety'])
X = d2.values

Y = labelencoder.fit_transform(Y)

from keras.utils import to_categorical
Y_binary = to_categorical(Y_train)

#Split the data into train and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

from keras.models import Sequential

from keras.layers import Dense

#initialize or create the neural network model
classifier = Sequential()

#create an input and hidden layer
classifier.add(Dense(output_dim=10,input_dim=16,activation="relu",init="uniform"))

#create a hidden layer
classifier.add(Dense(output_dim=10,activation="relu",init="uniform"))

#create an output layer
classifier.add(Dense(output_dim=4,activation="sigmoid",init="uniform"))

#compile the artificial neural network model
classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['categorical_accuracy'])

#Fit the train data into classifier
classifier.fit(X_train,Y_binary,batch_size=20,nb_epoch=100)

#Predict the values from classifier
Y_pred = classifier.predict(X_test)

#get the origial values from the probabilistic arrays of y_pred
predicted = np.argmax(Y_pred, axis=1)

#find the accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(predicted,Y_test)