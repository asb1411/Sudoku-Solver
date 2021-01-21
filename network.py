# import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import cv2

import numpy as np

import operator

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Activation

from keras.optimizers import RMSprop

from tensorflow.keras import backend as K

from tensorflow.keras.utils import to_categorical

cp = tf.compat.v1.ConfigProto()
cp.gpu_options.allow_growth = True

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

X_train = X_train.astype('float32')
X_train = X_train / 255.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0

model = Sequential()
model.add(Conv2D(128, (3, 3), kernel_initializer='glorot_uniform', input_shape=(28,28,1)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=64,epochs=10,verbose=1,validation_data=(X_test,y_test))
scores = model.evaluate(X_test,y_test,verbose=1)
print(scores)
model.save("model")
