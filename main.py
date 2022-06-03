import numpy as np
import os, glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
import cv2
from matplotlib import pyplot as plt
from load_image import load_image, show_train_history


dataPath = 'C:/Users/User/Desktop/final_project/used/Train'
# dataset, answer, labels = load_image(dataPath)
dataset, labels = load_image(dataPath)

x_train, x_valid, y_train, y_valid = train_test_split(dataset, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(36, 36, 3), activation='relu'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(85, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
train_history = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), validation_split=0.2, epochs=10, batch_size=128)
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')