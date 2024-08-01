import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras import Sequential # noqa
from keras.callbacks import EarlyStopping # noqa
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D # noqa
from keras.utils import to_categorical # noqa
import numpy as np # noqa
from scipy import io as sio # noqa


if __name__ == '__main__':
    num_classes = 26

    mat = sio.loadmat('../Datasets/emnist-letters.mat')
    data = mat['dataset']
    x_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]
    x_test = data['test'][0, 0]['images'][0, 0]
    y_test = data['test'][0, 0]['labels'][0, 0]
    x_train, x_test = x_train.reshape((x_train.shape[0], 28, 28, 1)), x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train, x_test = x_train / 255., x_test / 255.
    y_train, y_test = y_train - 1, y_test - 1
    y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)

    model = Sequential()

    model.add(Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(28, 28, 1)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=0)
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])
