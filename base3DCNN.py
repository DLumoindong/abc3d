#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm
import os


# In[ ]:


img_rows,img_cols,img_depth=100,100,15


# In[ ]:


X_tr=[]
path = 'ucf-50-10-sample/'

for categories in os.listdir(path):
    for vid in os.listdir(path+"/"+categories+"/"):
        vid = path+"/"+categories+"/"+vid
        frames = []
        cap = cv2.VideoCapture(vid)
        fps = cap.get(5)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {}".format(fps))

        for k in range(img_depth):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

            #plt.imshow(gray, cmap = plt.get_cmap('gray'))
            #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            #plt.show()
            #cv2.imshow('frame',gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        input=np.array(frames)

        print(input.shape)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        print(ipt.shape)

        X_tr.append(ipt)


# In[ ]:


X_tr_array = np.array(X_tr)   # convert the frames read into array

num_samples = len(X_tr_array)
print(num_samples)


# In[ ]:


label=np.ones((num_samples,),dtype = int)
label[0:49]= 0
label[50:99] = 1
label[100:149] = 2
label[150:199] = 3
label[200:249]= 4
label[250:299] = 5
label[300:349] = 6
label[350:399] = 7
label[400:449] = 8
label[449:] = 9


train_data = [X_tr_array,label]

(X_train, y_train) = (train_data[0],train_data[1])
print('X_train shape:', X_train.shape)

train_set = np.zeros((num_samples,img_rows,img_cols,img_depth, 1))

for h in range(num_samples):
    train_set[h,:,:,:,0]=X_train[h,:,:,:]

patch_size = img_depth    # img_depth or number of frames used for each video
print(train_set.shape, 'train samples')


# In[ ]:


train_set.shape


# In[ ]:


batch_size = 16
nb_classes = 10
nb_epoch = 10

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)


# number of convolutional filters to use at each layer
nb_filters = [32, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]

# Pre-processing

train_set = train_set.astype('float32')

train_set -= np.mean(train_set)

train_set /=np.max(train_set)

train_set


# In[ ]:


train_set.shape


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D, BatchNormalization, LeakyReLU)
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils, generic_utils


# In[ ]:

activate = LeakyReLU(alpha=0.3)
model = Sequential()

model.add(Conv3D(32, kernel_size=(3, 3, 3),input_shape=(img_rows, img_cols, patch_size, 1), activation='relu'))

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
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


# In[ ]:


# Split the data

X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)


# In[ ]:


hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
                 batch_size=batch_size,epochs = nb_epoch,shuffle=True)

score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])

