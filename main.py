"""
This is the main module responsible for solving the tasks.
To solve each task just run `python main.py`.
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn import metrics
# import dlib

import A1.landmark_MLP
# import src.acquiring
# import src.visualising
# import src.storing

basedir = '.\Datasets'
celeba_images_dir_train = os.path.join(basedir,'celeba')
celeba_images_dir_test = os.path.join(basedir,'celeba_test')
final_image_dir_train = os.path.join(celeba_images_dir_train,'img')
final_image_dir_test = os.path.join(celeba_images_dir_test,'img')
labels_filename = 'labels.csv'


def solve_A1_landmark_MLP():
    X, y = A1.landmark_MLP.extract_features_labels(final_image_dir = final_image_dir_train, celeba_images_dir = celeba_images_dir_train, labels_filename = labels_filename )
    X_test, y_test = A1.landmark_MLP.extract_features_labels(final_image_dir = final_image_dir_test, celeba_images_dir = celeba_images_dir_test, labels_filename = labels_filename )
    Y_test = np.array([y_test, -(y_test - 1)]).T
    Y = np.array([y, -(y - 1)]).T
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    # In this case, the test_size parameter is equal to 0.2, so that our validation set will
    # have 20% of the data, while the training set will have the remaining 80% of the data
    model = A1.landmark_MLP.create_MLP_model()
    print(model.summary())
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # Start the training and save its progress in a variable called 'history' and show the training time
    start_time = dt.datetime.now()
    print('Start learning with best params at {}'.format(str(start_time)))
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=250)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning time {}'.format(str(elapsed_time)))
    # Now that we have trained our model, save the plot of how our model performed
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_num = range(len(accuracy))
    plt.figure(1)
    plt.plot(epochs_num, accuracy, "b", label="Training accuracy")
    plt.plot(epochs_num, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend
    plt.savefig('Training and validation accuracy.png')

    plt.figure(2)
    plt.plot(epochs_num, loss, "b", label="Training loss")
    plt.plot(epochs_num, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('Training and validation loss.png')
    
    predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    predictions_sparse = []

    for one_hot_pred in predictions:
        if one_hot_pred[0] > one_hot_pred[1]:
            predictions_sparse.append(0)
        else:
            predictions_sparse.append(1)
    cm = metrics.confusion_matrix(y_test, predictions_sparse)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
    plt.figure(3)
    cm_display.plot()
    plt.title("Confusion matrix using landmark and MLP")
    plt.savefig('Confusion matrix using landmark and MLP')

def main():
    solve_A1_landmark_MLP()

if __name__ == "__main__":
    main()