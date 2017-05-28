'''Udacity-SDC Project 3 - Final'''
'''Imports'''
import os
import csv
import cv2
import keras
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *
from keras.models import Sequential, Model
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D

# Locate folder where the data is located
folder = './data/'
path = folder + 'driving_log.csv'

# I resized my images to allow faster training on my GPU. Previously, on the full size image, training took upwards of 20 minutes per model.
row_size = 16
col_size = 16


# Create a fucntion
def image_preprocessing(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(col_size, row_size))
    return resized


# Create a data loading function
def load_data(imgs, steering, folder, c_factor):
    tokens = []
    with open(path,'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            tokens.append(line)
    log_labels = tokens.pop(0)

    for i in range(len(tokens)):
        img_path = tokens[i][0]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        imgs.append(image_preprocessing(img))
        steering.append(float(tokens[i][3]))

    # Subtract the correction factor to the steering angle
    for i in range(len(tokens)):
        img_path = tokens[i][2]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        imgs.append(image_preprocessing(img))
        steering.append(float(tokens[i][3]) - c_factor)

    # Add correction factor to the steering angle
    for i in range(len(tokens)):
        img_path = tokens[i][1]
        img_path = folder+'IMG'+(img_path.split('IMG')[1]).strip()
        img = plt.imread(img_path)
        imgs.append(image_preprocessing(img))
        steering.append(float(tokens[i][3]) + c_factor)

# Create empty arrays to store the data
data={}
data['Images'] = []
data['Steering'] = []

# Apply the load_data() function and place it inside each of the data folders
load_data(data['Images'], data['Steering'], folder, 0.2)

# Place images into img_train as an np.array()
img_train = np.array(data['Images']).astype('float32')
# Place steering angles into steer_train as an np.array()
steer_train = np.array(data['Steering']).astype('float32')

# Create empty arrays to place flipped images into
aug_imgs = []
aug_steer = []

# Flip images and steering angles
aug_imgs = np.append(img_train, img_train[:,:,::-1], axis=0)
aug_steer = np.append(steer_train, -steer_train, axis=0)
img_train = np.append(img_train, images_train[:,:,::-1], axis=0)
steer_train = np.append(steer_train, -steer_train, axis=0)

# Input data into the training set
X_train, y_train = shuffle(img_train, steer_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

# Reshape all the training and validation sets
X_train = X_train.reshape(X_train.shape[0], row_size, col_size, 1)
X_val = X_val.reshape(X_val.shape[0], row_size, col_size, 1)

'''Create the model'''
# Define a sequential model
model = Sequential()
# Preprocess the data with a lambda layer
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(row_size,col_size,1)))
# Conv Layer 1
model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(row_size, col_size, 1), activation='elu'))
# Max Pooling 1
model.add(MaxPooling2D((4,4),(4,4),'valid'))
# Dropout 1
model.add(Dropout(0.25))
# Flatten Layer
model.add(Flatten())
# FC Layer
model.add(Dense(1))

# Using Adam optimizer for learning rate, Loss Function is MSE
model.compile(loss='mean_squared_error',optimizer='adam')

# Define batch size and epoch
batch_size=128
nb_epoch=9

# Fit the model
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val))

model.save("model12.h5")
