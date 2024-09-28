# -*- coding: utf-8 -*-
"""
#Script for DRN or DRNv2 training
Last modified on 9/26/2024
Written by Mingdi Liu
Contact: Mingdi Liu (DD1359406536@163.com)
"""

import scipy.io as scio
import numpy as np
import pandas as pd
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from sklearn.utils import shuffle
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from networks import *  # DRN and DRNv2


def read_mat_files(folder_path):
    """Read all .mat files from the specified folder."""
    mat_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".mat")]
    data = []
    for file_path in mat_files:
        try:
            mat_data = loadmat(file_path)
            data.append(mat_data)
        except Exception as e:
            print(f"Error reading file: {file_path}")
            print(f"Error details: {e}")
    return data


def read_values(folder_path, Mat_name, Label_name):
    """Extract specified matrices and labels from .mat files."""
    mat_data = read_mat_files(folder_path)

    # Print the read data for debugging
    for data in mat_data:
        print(data)

    size = np.shape(mat_data[0][Mat_name])
    length = len(mat_data)

    # Prepare data arrays
    try:
        data_extract = np.zeros([length, size[0], size[1], size[2]])
        label_extract = np.zeros([length])
    except:
        data_extract = np.zeros([length, size[0], size[1]])
        label_extract = np.zeros([length])

    # Extract data and labels
    for i in range(length):
        data_extract[i] = mat_data[i][Mat_name]
        label_extract[i] = mat_data[i][Label_name]

    return data_extract, label_extract


# Set folder path, Mat_name for training images, and Label_name for labels
X, Y = read_values(folder_path="./Train256/", Mat_name='D', Label_name='z')

# Select specific images from the dataset
#Mode
# X1 = X                            # All      9 images
# X1 = X[:, :, :, 0:2]                 # Adjacent 2 images
#X1 = X[:, :, :, [0, 1, 3, 5, 7]]  # Adjacent 5 images
X1 = X[:, :, :, [0, 2, 4, 6, 8]]  # Opposite 5 images

# Shuffle and scale labels
X, Y = shuffle(X1, Y, random_state=1337)
Y = Y * 1e6  # Convert meters to micrometers

# Set up distributed training strategy
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = DRN(input_shape=(256, 256, 5))
    # Alternatively use: model = DRNv2(input_shape=(256, 256, 5))

# Compile the model
adam = optimizers.Adam(learning_rate=1e-5)
model.compile(loss='mae', optimizer=adam, metrics=['mse'])

# Set up model checkpointing
model_checkpoint = ModelCheckpoint('model/Unet1.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

# Train the model
history = model.fit(x=X, y=Y, validation_split=0.3, batch_size=32, verbose=1, epochs=500, callbacks=[model_checkpoint])

# Plot training and validation loss
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
loss2 = history.history['mse']
val_loss2 = history.history['val_mse']

print("The minimum loss ", min(loss))
print('The minimum val_loss ', min(val_loss))
print('Epoch:', val_loss.index(min(val_loss)) + 1)

# Plot and save the training history
epochs = range(len(loss))
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./model/loss.png')

# Save results to CSV
results = np.zeros([len(loss), 4])
results[:, 0] = loss[:]
results[:, 1] = val_loss[:]
results[:, 2] = loss2[:]
results[:, 3] = val_loss2[:]

# Convert results to DataFrame and save
yuce = pd.DataFrame(results)
yuce.to_csv('./model/training_results.csv', header=False, index=False)





