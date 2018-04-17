
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
dataset_train = pd.read_csv('../dataset/Google_Stock_Price_Test.csv')

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
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], numberOfIndicators)))
regressor.add(Dropout(0.2))