#!/usr/bin/env python
# coding: utf-8


# IMPORTS
import cv2  # for capturing videos
import math  # for mathematical operations
import matplotlib.pyplot as plt  # for plotting the images
import pandas as pd
from keras.preprocessing import image  # for preprocessing the images
import numpy as np  # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize  # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os
from keras import backend as K
import tensorflow as tf
#from spark_tensorflow_distributor import MirroredStrategyRunner
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName('base3dcnn-dist').setMaster('spark://master:7077')
sc = SparkContext.getOrCreate(conf=conf)

#disable GPU due to RAPIDS error
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, LeakyReLU)
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils, generic_utils

mirrored_strategy = tf.contrib.distribute.MirroredStrategy()

# inputs for data prep
img_rows, img_cols, img_depth, img_channels = 96, 96, 25, 1
batch_size = 2
# nb_epoch = 20

fpath = 'ucf-50-10-sample/'


def data_prep(rows, cols, depth, channels, path):
    labels = []
    X_tr = []
    categories = os.listdir(path)
    nb_classes = len(categories)
    for category in categories:
        for vid in os.listdir(path + "/" + category + "/"):
            vid = path + "/" + category + "/" + vid
            frames = []
            cap = cv2.VideoCapture(vid)
            fps = cap.get(5)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {}".format(fps))

            for k in range(depth):
                ret, frame = cap.read()
                frame = cv2.resize(frame, (rows, cols), interpolation=cv2.INTER_AREA)
                if channels == 1:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
                else:
                    frames.append(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

            input = np.array(frames)

            print(input.shape)
            ipt = np.rollaxis(np.rollaxis(input, 2, 0), 2, 0)
            print(ipt.shape)

            X_tr.append(ipt)
            X_tr_array = np.array(X_tr)  # convert the frames read into array
            num_samples = len(X_tr_array)
            print(num_samples)
            labels.append(categories.index(category))

    train_data = [X_tr_array, labels]

    (X_train, y_train) = (train_data[0], train_data[1])

    train_set = np.zeros((num_samples, img_rows, img_cols, img_depth, channels))

    for h in range(num_samples):
        train_set[h, :, :, :, 0] = X_train[h, :, :, :]

    patch_size = img_depth  # img_depth or number of frames used for each video

    train_set.shape

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    # number of convolutional filters to use at each layer
    nb_filters = [32, 32]

    # level of pooling to perform at each layer (POOL x POOL)
    nb_pool = [3, 3]

    # level of convolution to perform at each layer (CONV x CONV)
    nb_conv = [5, 5]

    # Pre-processing

    train_set = train_set.astype('float32')

    train_set -= np.mean(train_set)

    train_set /= np.max(train_set)

    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

    return X_train_new, X_val_new, y_train_new, y_val_new, nb_classes


def base3dcnn(leaky_relu_alpha, learn_rate, rows, cols, depth, channels, classes):
    activate = LeakyReLU(alpha=0.4)
    model = Sequential()

    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(rows, cols, depth, 1), activation='relu'))

    model.add(Activation(activate))
    model.add(Conv3D(32, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation(activate))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation(activate))
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation(activate))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation(activate))
    model.add(Conv3D(64, padding="same", kernel_size=(3, 3, 3)))
    model.add(Activation(activate))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation=activate))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.0001),
                  metrics=['accuracy'])
    return model


def trainbase(rows, cols, depth, channels, leaky_relu_alpha, learn_rate, batch_size, num_epochs, trainx, trainy, valx, valy, classes):
    model = base3dcnn(leaky_relu_alpha, learn_rate, rows, cols, depth, channels, classes)

    hist = model.fit(trainx, trainy, validation_data=(valx, valy),
                     batch_size=batch_size, epochs=num_epochs, shuffle=True)

    score = model.evaluate(valx, valy, batch_size=batch_size)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    print(model.output.op.name)
    K.set_learning_phase(0)
    saver = tf.train.Saver()
    saver.save(K.get_session(), '/home/deo/kuliah/thesis/base3DCNN_model.ckpt')


x_train, x_val, y_train, y_val, nb_classes = data_prep(img_rows, img_cols, img_depth, img_channels, fpath)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

with sess:
    sess.run(trainbase(img_rows, img_cols, img_depth, img_channels, 0.4, 0.0001, batch_size, 20, x_train, y_train, x_val, y_val, 10))
