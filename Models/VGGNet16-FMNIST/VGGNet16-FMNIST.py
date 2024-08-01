import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras import Sequential # noqa
from keras.datasets import fashion_mnist # noqa
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D # noqa
from keras.utils import to_categorical # noqa
from PIL import Image # noqa
import numpy as np # noqa


if __name__ == '__main__':
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train_resized = np.array([np.array(Image.fromarray(img).resize((56, 56))) for img in x_train])
    x_test_resized = np.array([np.array(Image.fromarray(img).resize((56, 56))) for img in x_test])
    x_train_resized, x_test_resized = np.expand_dims(x_train_resized, -1), np.expand_dims(x_test_resized, -1)
    x_train, x_test = x_train_resized / 255., x_test_resized / 255.
    y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(56, 56, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
