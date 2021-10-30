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
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils
from binarizedModel import *

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

            # print(input.shape)
            ipt = np.rollaxis(np.rollaxis(input, 2, 0), 2, 0)
            # print(ipt.shape)

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


testModel = Sequential([
    Binarized3D(32, 3, 3, 3, 1, 1, 1, padding='VALID', bias=True, reuse=False, name='B3D_1'),
    BatchNormalization(),
    HardTanh(),
    Binarized3D(32, 3, 3, 3, 1, 1, 1, padding='SAME', bias=True, reuse=False, name='B3D_1'),
    SpatialTemporalMaxPooling(3, 3, 3, 1, 1, 1),
    BatchNormalization(),
    HardTanh(),
    Binarized3D(64, 3, 3, 3, 1, 1, 1, padding='SAME', bias=True, reuse=False, name='B3D_1'),
    SpatialTemporalMaxPooling(3, 3, 3, 1, 1, 1),
    BatchNormalization(),
    HardTanh(),
    Binarized3D(64, 3, 3, 3, 1, 1, 1, padding='SAME', bias=True, reuse=False, name='B3D_1'),
    SpatialTemporalMaxPooling(3, 3, 3, 1, 1, 1),
    BatchNormalization(),
    HardTanh(),
    BinarizedAffine(512, bias=False),
    BatchNormalization(),
    HardTanh(),
    BinarizedAffine(10),
    BatchNormalization()
])

x_train, x_val, y_train, y_val, nb_classes = data_prep(img_rows, img_cols, img_depth, img_channels, fpath)
print("Training data shape: ", x_train.shape)
print("Validation data_shape: ", x_val.shape)
print("\n====================================================")
print("Training Started")
print("====================================================\n")

train(testModel, x_train,
      batch_size=8,
      checkpoint_dir='/chkpt/',
      log_dir='/logs/',
      num_epochs=10)
