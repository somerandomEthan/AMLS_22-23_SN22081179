"""
This is the module dealing with task A1 usign landmark extraction and multilayer perceptron
It contains model to do the classificaiton and model optimization.
"""

import os
import numpy as np
import cv2
import dlib

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image

def extract_features_labels(final_image_dir, celeba_images_dir, labels_filename):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extract the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(final_image_dir, l) for l in os.listdir(final_image_dir)]
    target_size = None
    labels_file = open(os.path.join(celeba_images_dir, labels_filename), 'r')
    lines = labels_file.readlines()
    gender_labels = dict((line.split('\t')[1], int(line.split('\t')[3])) for line in lines[1:])
    if os.path.isdir(celeba_images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name = img_path.split('.')[1].split('\\')[-1]
            label_file_name = file_name + '.jpg'

            # load image
            img = img_to_array(
                load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(gender_labels[label_file_name])

    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels) + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    return landmark_features, gender_labels

def create_MLP_model(optimizer='Adam', learn_rate=0.001, amsgrad=False):

    num_classes = 2
    model = Sequential()
    # Add the input layer and flatten the input
    model.add(Flatten(input_shape = ( 68, 2)))
    # Add the first hidden layer
    model.add(Dense(units=128,activation='relu'))
    # Add the second hidden layer
    model.add(Dense(units=64,activation='relu'))
    # Add the output layer
    model.add(Dense(units=num_classes, activation='sigmoid'))

    optimizer = Adam(lr=learn_rate, amsgrad=amsgrad )
    # Compile model (sparse cross-entropy can be used if one hot encoding not used)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.mae, keras.metrics.categorical_accuracy])

    return model


