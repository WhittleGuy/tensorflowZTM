#!/usr/bin/python3

import zipfile
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import wget

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download the zip file of images
wget.download(
    'https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip')
zip_ref = zipfile.ZipFile('pizza_steak.zip', 'r')
zip_ref.extractall()
zip_ref.close()

# Define directories
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"

# Set up datagens
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=20,
                                             shear_range=0.2,
                                             zoom_range=0.2,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1/255.)

# Generate data
train_data = train_datagen_augmented.flow_from_directory(train_dir,
                                                         target_size=(
                                                             224, 224),
                                                         batch_size=32,
                                                         class_mode='binary',
                                                         shuffle=True)

test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode='binary')

# Create a model
model = Sequential([
    Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(10, 3, activation='relu'),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# Fit the model
history = model.fit(train_data,
                    epochs=1,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=len(test_data))

model.save('cnn_test_model.h5')
