#!/usr/bin/env python
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import dataset as ds
import numpy as np
import sys


bin = np.arange(256)
if not ds.check_folder():
    ds.makeDataset(bin,type="s")

x,y =ds.readDataset(bin,type="s")

#train_images, train_labels,test_images,test_labels =ds.readDataset(bin,type="s")
order = np.random.permutation(y.shape[0])
x = x[order, :, :, :]
y = y[order, :]
div = int(0.7*y.shape[0])

train_images, test_images = x[0:div,:,:,:] / 1.0, x[div:,:,:,:] / 1.0
train_labels, test_labels = y[0:div,:], y[div:,:]

"""
if "-v" in sys.argv:
    class_names = ['Healthy', 'Parkinson']

    plt.figure(figsize=(10,10))
    for i in range(26):
        plt.subplot(5,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)

        plt.xlabel(class_names[test_labels[i][0]])
    plt.show()
"""
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

model = models.Sequential()
#model.add(data_augmentation)
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(8, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)