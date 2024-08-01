import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np # noqa
from keras import Sequential # noqa
from keras.layers import Dense # noqa
from sklearn.model_selection import train_test_split # noqa
from sklearn.preprocessing import MinMaxScaler # noqa
from sklearn.metrics import mean_absolute_error # noqa


if __name__ == '__main__':
    x = np.linspace(0, 4 * np.pi, 10000)
    y = np.column_stack((np.sin(x), np.cos(x), np.sqrt(x)))
    scaler = MinMaxScaler()
    y = scaler.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    model = Sequential()

    model.add(Dense(64, input_shape=(1,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='linear'))
    
    model.compile(optimizer='adam', loss='mae')
    
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
