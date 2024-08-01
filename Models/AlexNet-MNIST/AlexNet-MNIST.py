import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras import Sequential # noqa
from keras.datasets import mnist # noqa
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D # noqa
from keras.utils import to_categorical # noqa
from keras.callbacks import EarlyStopping # noqa
import numpy as np # noqa


if __name__ == '__main__':
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255., x_test / 255.
    x_train, x_test = x_train.reshape((len(x_train), 28, 28, 1)), x_test.reshape((len(x_test), 28, 28, 1))
    y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=0)
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])

