"""
This is the module dealing with task A1 usign Convolutional Neural Network(CNN)
It contains model to do the classificaiton and model optimization.
"""

import os
import numpy as np
import cv2
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.optimizers import Adam

def extract_images_labels(final_image_dir, celeba_images_dir, labels_filename):
    """
    This funtion changes all images into numpy array.
    It also extract the gender label for each image.
    :return:
        images_datasets:  an array containing 128 *128 pixels with 3 RGB channels for each image
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(final_image_dir, l) for l in os.listdir(final_image_dir)]
    labels_file = open(os.path.join(celeba_images_dir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = dict((line.split('\t')[1], int(line.split('\t')[2])) for line in lines[1:])
    if os.path.isdir(celeba_images_dir):
        all_images = []
        all_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('\\')[-1]
            label_file_name = file_name + '.jpg'

            # load image
            # image grayscaling at import(if using 0 to read image then grayscale mode)
            img = cv2.imread(img_path)
            # rescaling image, the input size of the CNN is 128 * 128 *3
            res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
             # turn to float for zero centering
            res = res.astype(float)
            if res is not None:
                all_images.append(res)
                all_labels.append(gender_labels[label_file_name])


    images_datasets = np.array(all_images)
    gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return images_datasets, gender_labels

def create_CNN_model(optimizer='adam', activation=tf.nn.leaky_relu):

    num_classes = 2
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='SAME', activation=activation,  input_shape=(128, 128, 3), data_format="channels_last"))
    model.add(Dropout(0.2))
    model.add(Conv2D(16, (3, 3), padding='SAME', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Dropout(0.2))
    model.add(Conv2D(8, (3, 3), padding='SAME', activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model (sparse cross-entropy can be used if one hot encoding not used)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

    return model