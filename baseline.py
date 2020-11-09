#!/usr/bin/env python
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import dataset as ds
import numpy as np
import sys

bin = np.arange(256)
x, y =ds.makeDataset(bin,type="s")

order = np.random.permutation(y.shape[0])
x = x[order, :, :, :]
y = y[order, :]
div = int(0.75*y.shape[0])

train_images, test_images = x[0:div,:,:,:] / 255.0, x[div:,:,:,:] / 255.0
train_labels, test_labels = y[0:div,:], y[div:,:]

if "-v" in sys.argv:
    class_names = ['Healthy', 'Parkinson']

    plt.figure(figsize=(10,10))
    for i in range(102):
        plt.subplot(5,21,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

model = models.Sequential()
model.add(layers.Conv2D(8, (11, 11), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (7, 7), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)