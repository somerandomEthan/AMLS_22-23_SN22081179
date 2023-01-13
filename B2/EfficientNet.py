import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam
from keras.applications.efficientnet_v2 import EfficientNetV2B0

def extract_images_labels(final_image_dir, cartoon_images_dir, labels_filename):

    image_paths = [os.path.join(final_image_dir, l) for l in os.listdir(final_image_dir)]
    labels_file = open(os.path.join(cartoon_images_dir, labels_filename), 'r')
    lines = labels_file.readlines()
    individual_labels = dict((line.split('\t')[3].strip('\n') , int(line.split('\t')[1])) for line in lines[1:])# the label for eye color is located at the second column
    if os.path.isdir(cartoon_images_dir):
        all_images = []
        all_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('\\')[-1]
            label_file_name = file_name + '.png'

            # load image
            img = cv2.imread(img_path)
            # rescaling image to reduce size
            res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
             # turn to float for zero centering
            res = res.astype(float)
            if res is not None:
                all_images.append(res)
                all_labels.append(individual_labels[label_file_name])


    images_datasets = np.array(all_images)
    eyecolor_labels = np.array(all_labels) 
    return images_datasets, eyecolor_labels

def one_hot_encoder(labels):
    label_encoder = LabelBinarizer()
    label_encoder.fit(labels.reshape(-1, 1))
    encoded_labels = label_encoder.transform(labels.reshape(-1, 1))
    return encoded_labels

def create_EfficientNetV2B0_model(optimizer='adam', learn_rate=0.001, amsgrad=False,):
    num_classes = 5
    base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_tensor=None, input_shape=(128, 128, 3), pooling=None, classifier_activation='softmax')
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=tf.nn.leaky_relu))
    model.add(Dense(num_classes, activation='sigmoid'))


    optimizer = Adam(learning_rate=learn_rate, amsgrad=amsgrad )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])
    return model