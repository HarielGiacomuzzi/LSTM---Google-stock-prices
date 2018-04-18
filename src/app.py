
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

numberOfPastSteps = 60
numberOfIndicators = 1

# Import the training set
dataset_train = pd.read_csv('/Users/hariel.dias/Desktop/Hariel/Deep Learning/RNN/dataset/Google_Stock_Price_Train.csv')

# gets only the opening prices and returns as a np array
training_set = dataset_train.iloc[:, 1:2].values

# create a scaler that will apply normalization to our dataset, making the values range from 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scaler.fit_transform(training_set)

# create the data structures with the number of past time steps and the number of future predicted steps
x_train = []
y_train = []

for i in range(numberOfPastSteps, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-numberOfPastSteps:i, 0])
    y_train.append(training_set_scaled[i, 0])

# transforms the datastructures to np arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape the arrays for the correct input as we can see in the input shapes section of Keras docs of recurrent layers
# (batch_size, timesteps, input_dims) --> (number of observations, timesteps, indicators)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], numberOfIndicators))


# Building the RNN
regressor = Sequential()

# return_sequences is set since we are adding another stack of LSTM in the output of this layer
# input_shape only needs to know the time steps and the number of indicators, the other dimension is automatically expected
regressor.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], numberOfIndicators)))
regressor.add(Dropout(0.2))

# layer 2
regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(0.2))

# layer 3
regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(0.2))

# layer 3
regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(0.2))

# layer 4
# since the next layer is not a LSTM then the return_sequences is not needed
regressor.add(LSTM(units=60, return_sequences=False))
regressor.add(Dropout(0.2))

# output layer
regressor.add(Dense(units=1))

# compile the model
regressor.compile(optimizer='adam', loss='mean_squared_error')

# epochs --> number of times it will go throug the entire training set
# batch_size --> number of observations befor updating the weights
regressor.fit(x_train,y_train, epochs=120, verbose=1, batch_size=32)

#
# Visualizing results
#

# Getting the real values of the Stock prices
dataset_test = pd.read_csv('/Users/hariel.dias/Desktop/Hariel/Deep Learning/RNN/dataset/Google_Stock_Price_Test.csv')
# gettign only the open value
real_data = dataset_test.iloc[:, 1:2].values

# Getting the predicted values for the 30 next steps
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - numberOfPastSteps:].values

# set it to the shape of lines and one column
inputs = inputs.reshape(-1, 1)

# scale the inputs as we scaled for training...
# remember that part of it is already transformed, so just need to use the transform insted of the fit_transform
inputs = scaler.transform(inputs)

# set it to the correct 3D structure
x_test = []

for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# getting the predictions finally
preds = regressor.predict(x_test)

# now inverse the scaled values to the "normal" values
preds = scaler.inverse_transform(preds)

#
# Creating a graph with the real values and the predicted ones
#

plt.plot(real_data, color='blue', label='Real Data')
plt.plot(preds, color='orange', label='Predicted Data')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()







