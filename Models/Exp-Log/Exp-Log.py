import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np # noqa
from keras import Sequential # noqa
from keras.layers import Dense # noqa
from sklearn.model_selection import train_test_split # noqa
from sklearn.preprocessing import MinMaxScaler # noqa


if __name__ == '__main__':
    x = np.linspace(0.1, 4.0, 10000)
    y = np.column_stack((np.exp(x), np.log(x)))
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    model = Sequential()
    model.add(Dense(64, input_shape=(1,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    model.fit(x_train, y_train, epochs=128, validation_split=0.1)
