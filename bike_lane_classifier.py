from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

train_dir = './TRAINING_DATA'
test_dir = './TEST_DATA'

img_size = 120
LR = 1e-3

model_name = 'bike_lane--{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    if img.split('_')[0] == 'BIKE':
        return [1, 0]
    elif img.split('_')[0] == 'NOT':
        return [0, 1]

def create_training_data():
    train_data = []
    
    for img in os.listdir(train_dir):
        label = label_img(img)
        path = os.path.join(train_dir, img)

        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((img_size, img_size), Image.ANTIALIAS)

        train_data.append( [np.array(img), np.array(label)] )
    shuffle(train_data) # Mixes up data so that a dataset doesn't contain all bike lanes or all not bike lanes
    np.save('train_data.npy', train_data)
    return train_data

def process_test_data():
    test_data = []

    for img in os.listdir(test_dir):
        path = os.path.join(test_dir, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((img_size, img_size), Image.ANTIALIAS)

            test_data.append( [np.array(img), np.array(img.split('_')[2])] )

    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data

train_data = create_training_data()
plt.imshow(train_data[5][0], cmap = 'gist_gray')
plt.show()
print(train_data[5][1])


# Build the conv network
convnet = input_data(shape=[None, img_size, img_size, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=3)


# Data Splitting