#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
from keras import models,layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.applications import VGG16
from keras import optimizers
import keras
import matplotlib.pyplot as plt

conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150,150,3))

base_dir = '/home/peichao/Desktop/python_experiment/all/cats_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

test_generator = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_feature(directory, sample_count):
    features = np.zeros(shape=(sample_count,4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = test_generator.flow_from_directory(directory, target_size=(150,150), batch_size=batch_size, class_mode='binary')
    i=0
    for input_batchs, input_label in generator:
        features_batch = conv_base.predict(input_batchs)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size]=input_label
        i+=1
        if i*batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels = extract_feature(train_dir, 2000)
validation_features, validation_labels = extract_feature(validation_dir, 1000)
test_features, test_labels = extract_feature(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features,(1000, 4*4*512))

network = models.Sequential()
#network.add(conv_base)
#network.add(layers.Flatten())
network.add(layers.Dense(512, activation = 'relu', input_dim=4*4*512))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(1, activation = 'sigmoid'))
network.compile(optimizer = optimizers.RMSprop(2e-5), loss='binary_crossentropy', metrics=['acc'])

callback_list = [keras.callbacks.EarlyStopping(monitor = 'acc', patience=1,), 
                keras.callbacks.ModelCheckpoint(filepath='cats_and_dogs_small_2.h5', monitor='val_loss', save_best_only=True,),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,)]

network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#network.fit(x,y, epochs=10, callbacks=callback_list, validation_data=(x_val, y_val))


history = network.fit(train_features, train_labels, epochs=30, callbacks=callback_list, batch_size=32, validation_data=(validation_features, validation_labels))

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(acc)+1)

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('accuracy diagram')
plt.legend()
plt.show()