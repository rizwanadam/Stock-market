
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Part-1
#importing training set
dataset_train = pd.read_csv("Training_set.csv")
training_set = dataset_train.iloc[:,1:6].values

#print(training_set[:,4])
for x in training_set[:,4]:
    x=x[:-1]
    x=float(x)
    x=x*1000000

#feature scaling using normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
trainingset_scaled = sc.fit_transform(training_set)
 
#creating a datstructure with 60 timesteps and 1 output.
#60 timesteps means that the at any time t the rnn with look at the trend from 60 days back
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(trainingset_scaled[i-60:i,0])
    y_train.append(trainingset_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)  

#reshaping. to add another dimension ie the indicator/s
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Part-2 Building the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
#initializing the rnn
regressor = Sequential()

#adding the 1st lstm layer and some dropout regularization 
#units refers to number of neurons in the hidden layer
#neurons are 50 to maintain high dimentionality
regressor.add(LSTM(units = 50,return_sequences = True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

#adding the 2st lstm layer and some dropout regularization. 
#No need to specify input_shape since it is automatically recognized
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#adding the 3st lstm layer and some dropout regularization
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))
 
#adding the 4st lstm layer and some dropout regularization 
#return_sequences  false. no need to return any sequences for the last layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1))

#Compiling the rnn.
#The loss function will mean squared error since this a regression problem
regressor.compile(optimizer = 'adam',loss="mean_squared_error")

#fitting the rnn to the training set
regressor.fit(x_train,y_train,epochs=100, batch_size =32)

#Part-3 Making the predictions and visualising the results
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
#getting the predicted stock prices
#for vertical concatenation use axis =0
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price=regressor.predict(x_test)
#inverse the scale to get the actual prices 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price,color = 'red',label ='Real Google stock price(Jan 2017)')
plt.plot(predicted_stock_price,color = 'blue',label ='Predicted Google stock price(Jan 2017)')
plt.title("Google stock price prediction")
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()



