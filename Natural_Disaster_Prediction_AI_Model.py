#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ## 1. Importing Libraries

# In[2]:

import tensorflow as tf

from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import legacy

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import pandas as pd


# In[3]:

for device in tf.config.list_physical_devices():
    print(": {}".format(device.name))


# ## 2. Gathering Data

# In[4]:


# defining paths of train, validation and test data
train_path = r"C:\Users\asus\OneDrive\Desktop\wildfire\train"
valid_path = r"C:\Users\asus\OneDrive\Desktop\wildfire\valid"
test_path = r"C:\Users\asus\OneDrive\Desktop\wildfire\test"


# In[5]:

image_shape = (350,350,3)
N_CLASSES = 2
BATCH_SIZE = 256

train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
train_generator = train_datagen.flow_from_directory(train_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')

valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
valid_generator = valid_datagen.flow_from_directory(valid_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')

test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')


# ## 3. Building the model

# In[6]:

weight_decay = 1e-3

first_model = Sequential([
    Conv2D(filters = 8 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 16 , kernel_size = 2, activation = 'relu', 
    input_shape = image_shape), MaxPooling2D(pool_size = 2),
    
    Conv2D(filters = 32 , kernel_size = 2, activation = 'relu',
           kernel_regularizer = regularizers.l2(weight_decay)),
    MaxPooling2D(pool_size = 2),
    
    Dropout(0.4),
    Flatten(),
    Dense(300,activation='relu'),
    Dropout(0.5),
    Dense(2,activation='softmax')
])
first_model.summary()


# ## 4. Training the model

# In[7]:

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

checkpointer = ModelCheckpoint('first_model.hdf5',verbose=1, save_best_only= True)
early_stopping = EarlyStopping(monitor= 'val_loss', patience= 10)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
first_model.compile(loss= 'categorical_crossentropy', optimizer= optimizer,
                    metrics=['AUC','acc'])
  
history = first_model.fit(train_generator,
                    epochs = 50,
                    verbose = 1,
                    validation_data = valid_generator,
                    callbacks = [checkpointer, early_stopping])


# In[8]:

plt.plot(history.history['acc'], label = 'train',)
plt.plot(history.history['val_acc'], label = 'valid')

plt.legend(loc = 'lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.show()


# In[10]:

plt.hist(history.history['acc'], label = 'train',color='green')

plt.legend(loc = 'lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy')


# In[11]:

plt.hist(history.history['val_acc'], label = 'valid', color='orange')
plt.legend(loc = 'lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy') 


# In[12]:

plt.hist(history.history['acc'], label = 'train',color='green')
plt.hist(history.history['val_acc'], label = 'valid', color='orange')

plt.legend(loc = 'lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy')


# In[9]:

result = first_model.evaluate(test_generator)


# In[ ]:




