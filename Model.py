import numpy as np
import os
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = 'Cats_and_Dogs_filtered'
train_dir = path+'/train'
train_cats_dir = train_dir+'/cats'
train_dogs_dir = train_dir+'/dogs'

total_train = len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))

train_image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, height_shift_range = 0.1, width_shift_range = 0.1,
                                     rotation_range = 40, zoom_range = 0.2)

train_data_gen = train_image_gen.flow_from_directory(batch_size = 10, directory = train_dir, shuffle = 1,
                                                             target_size = (32, 32), class_mode = 'binary')

model = Sequential()
model.add(Conv2D(64, 3, input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_data_gen, steps_per_epoch = total_train//10, epochs = 10)

model.save('Dogs_Cats_Model.h5')