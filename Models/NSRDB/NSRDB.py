import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys # noqa
import pandas as pd # noqa
import numpy as np # noqa
from keras import Sequential # noqa
from keras.layers import Dense # noqa
from sklearn.model_selection import train_test_split # noqa
from sklearn.preprocessing import StandardScaler # noqa
from sklearn.metrics import mean_absolute_error # noqa


if __name__ == '__main__':
    # Load data with your account ...

    # Select relevant columns
    X = df[['GHI', 'DHI', 'DNI', 'Relative Humidity', 'Temperature', 'Solar Zenith Angle']]
    y = df[['Wind Speed', 'Wind Direction']]

    # Normalize the input features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Normalize the output labels
    scaler_y = StandardScaler()
    y_normalized = scaler_y.fit_transform(y)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.1, random_state=42)

    # Define the neural network architecture
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))
