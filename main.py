"""
This is the main module responsible for solving the tasks.
To solve each task just run `python main.py`.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

import A1.landmark_MLP
import A1.CNN
import A1.landmark_XGBoost
import A2.landmark_MLP
import B1.VGG
import B1.EfficientNet
import B2.EfficientNet
import B2.EfficientNet_Cropping


basedir = '.\Datasets'
celeba_images_dir_train_A = os.path.join(basedir,'celeba')
celeba_images_dir_test_A = os.path.join(basedir,'celeba_test')
final_image_dir_train_A = os.path.join(celeba_images_dir_train_A,'img')
final_image_dir_test_A = os.path.join(celeba_images_dir_test_A,'img')
labels_filename = 'labels.csv'
cartoon_images_dir_train = os.path.join(basedir,'cartoon_set')
cartoon_images_dir_test = os.path.join(basedir,'cartoon_set_test')
final_cartoon_image_dir_train = os.path.join(cartoon_images_dir_train,'img')
final_cartoon_image_dir_test = os.path.join(cartoon_images_dir_test,'img')


def solve_A1_landmark_XGBoost():
    X, y = A1.landmark_XGBoost.extract_features_labels(final_image_dir = final_image_dir_train_A, celeba_images_dir = celeba_images_dir_train_A, labels_filename = labels_filename )
    X_test, y_test = A1.landmark_XGBoost.extract_features_labels(final_image_dir = final_image_dir_test_A, celeba_images_dir = celeba_images_dir_test_A, labels_filename = labels_filename )
    model = A1.landmark_XGBoost.create_XGBoost_model()
    model.fit(X, y)
    y_pred = model.predict(X_test)
    print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    # grid serch
    params = {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5]}
    FOLDS = 3
    PARAM_COMB = 5
    skf = StratifiedKFold(n_splits=FOLDS, shuffle = True, random_state = 1001)
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=PARAM_COMB, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001 )
    start_time = A1.landmark_XGBoost.timer(None) # timing starts from this point for "start_time" variable
    random_search.fit(X, y)
    A1.landmark_XGBoost.timer(start_time) # timing ends here for "start_time" variable
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (FOLDS, PARAM_COMB))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)

    best_model = random_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    print('XGBoost best model with grid search accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_best)))

    cm = metrics.confusion_matrix(y_test, y_pred_best)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
    plt.figure(3)
    cm_display.plot()
    plt.title("Confusion matrix using landmark and XGBoost")
    plt.savefig('A1 Confusion matrix using landmark and XGBoost')

def solve_A1_landmark_MLP():
    X, y = A1.landmark_MLP.extract_features_labels(final_image_dir = final_image_dir_train_A, celeba_images_dir = celeba_images_dir_train_A, labels_filename = labels_filename )
    X_test, y_test = A1.landmark_MLP.extract_features_labels(final_image_dir = final_image_dir_test_A, celeba_images_dir = celeba_images_dir_test_A, labels_filename = labels_filename )
    Y_test = np.array([y_test, -(y_test - 1)]).T
    Y = np.array([y, -(y - 1)]).T
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    # In this case, the test_size parameter is equal to 0.2, so that our validation set will
    # have 20% of the data, while the training set will have the remaining 80% of the data
    

    # Retrive the model
    model = A1.landmark_MLP.create_MLP_model()
    print(model.summary())
    BATCH_SIZE = 100
    SHUFFLE_BUFFER_SIZE = 100
    EPOCHS = 20
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # Start the training and save its progress in a variable called 'history' and show the training time
    start_time = dt.datetime.now()
    print('Start learning with best params at {}'.format(str(start_time)))
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=EPOCHS)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning time {}'.format(str(elapsed_time)))
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)
    # Now that we have trained our model, save the plot of how our model performed
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_num = range(len(accuracy))
    plt.figure(1)
    plt.plot(epochs_num, accuracy, "b", label="Training accuracy")
    plt.plot(epochs_num, val_accuracy, "r", label="Validation accuracy")
    plt.title("A1 Training and validation accuracy for landmarks and MLP")
    plt.legend
    plt.savefig('A1 Training and validation accuracy for landmarks and MLP.png')

    plt.figure(2)
    plt.plot(epochs_num, loss, "b", label="Training loss")
    plt.plot(epochs_num, val_loss, "r", label="Validation loss")
    plt.title("A1 Training and validation loss for landmarks and MLP")
    plt.legend()
    plt.savefig('A1 Training and validation loss for landmarks and MLP.png')
    
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
    plt.title("A1 Confusion matrix using landmark and MLP")
    plt.savefig('A1 Confusion matrix using landmark and MLP')



def solve_A1_CNN():
    X, y = A1.CNN.extract_images_labels(final_image_dir = final_image_dir_train_A, celeba_images_dir = celeba_images_dir_train_A, labels_filename = labels_filename )
    X_test, y_test = A1.CNN.extract_images_labels(final_image_dir = final_image_dir_test_A, celeba_images_dir = celeba_images_dir_test_A, labels_filename = labels_filename )
    Y_test = np.array([y_test, -(y_test - 1)]).T
    Y = np.array([y, -(y - 1)]).T
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    # In this case, the test_size parameter is equal to 0.2, so that our validation set will
    # have 20% of the data, while the training set will have the remaining 80% of the data

    model = A1.CNN.create_CNN_model()
    model.summary()
    BATCH_SIZE = 60
    SHUFFLE_BUFFER_SIZE = 100
    EPOCHS = 20
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    # Start the training and save its progress in a variable called 'history'
    start_time = dt.datetime.now()
    print('Start learning with best params at {}'.format(str(start_time)))
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=EPOCHS)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning time {}'.format(str(elapsed_time)))
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)

    # Now that we have trained our model, save the plot of how our model performed
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_num = range(len(accuracy))
    plt.figure(1)
    plt.plot(epochs_num, accuracy, "b", label="Training accuracy")
    plt.plot(epochs_num, val_accuracy, "r", label="Validation accuracy")
    plt.title("A1 Training and validation accuracy for landmarks and MLP")
    plt.legend
    plt.savefig('A1 Training and validation accuracy for landmarks and MLP.png')

    plt.figure(2)
    plt.plot(epochs_num, loss, "b", label="Training loss")
    plt.plot(epochs_num, val_loss, "r", label="Validation loss")
    plt.title("A1 Training and validation loss for CNN")
    plt.legend()
    plt.savefig('A1 Training and validation loss for CNN.png')
    
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
    plt.title("A1 Confusion matrix using CNN")
    plt.savefig('A1 Confusion matrix using CNN')


def solve_A2_MLP():
    X, y = A2.landmark_MLP.extract_features_labels(final_image_dir = final_image_dir_train_A, celeba_images_dir = celeba_images_dir_train_A, labels_filename = labels_filename )
    X_test, y_test = A2.landmark_MLP.extract_features_labels(final_image_dir = final_image_dir_test_A, celeba_images_dir = celeba_images_dir_test_A, labels_filename = labels_filename )
    Y_test = np.array([y_test, -(y_test - 1)]).T
    Y = np.array([y, -(y - 1)]).T
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    # In this case, the test_size parameter is equal to 0.2, so that our validation set will
    # have 20% of the data, while the training set will have the remaining 80% of the data
    
    # Retrive the model
    model = A2.landmark_MLP.create_MLP_model()
    print(model.summary())
    BATCH_SIZE = 100
    SHUFFLE_BUFFER_SIZE = 100
    EPOCHS = 100
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # Start the training and save its progress in a variable called 'history' and show the training time
    start_time = dt.datetime.now()
    print('Start learning with best params at {}'.format(str(start_time)))
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=EPOCHS)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time= end_time - start_time
    print('Elapsed learning time {}'.format(str(elapsed_time)))
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)
    # Now that we have trained our model, save the plot of how our model performed
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_num = range(len(accuracy))
    plt.figure(1)
    plt.plot(epochs_num, accuracy, "b", label="Training accuracy")
    plt.plot(epochs_num, val_accuracy, "r", label="Validation accuracy")
    plt.title("A2 Training and validation accuracy for landmarks and MLP")
    plt.legend
    plt.savefig('A2 Training and validation accuracy for landmarks and MLP.png')

    plt.figure(2)
    plt.plot(epochs_num, loss, "b", label="Training loss")
    plt.plot(epochs_num, val_loss, "r", label="Validation loss")
    plt.title("A2 Training and validation loss for landmarks and MLP")
    plt.legend()
    plt.savefig('A2 Training and validation loss for landmarks and MLP.png')
    
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
    plt.title("A2 Confusion matrix using landmark and MLP")
    plt.savefig('A2 Confusion matrix using landmark and MLP')


def solve_B1_VGG():
    X, y = B1.VGG.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    X_test, y_test = B1.VGG.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    Y_test = B1.VGG.one_hot_encoder(np.array(y_test))
    Y = B1.VGG.one_hot_encoder(np.array(y))
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    model = B1.VGG.create_VGG19_model()
    model.summary()
    BATCH_SIZE = 1000
    SHUFFLE_BUFFER_SIZE = 100
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=40)
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(accuracy))

    plt.figure(1)
    plt.plot(epochs, accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend
    plt.savefig('B1 Training and validation accuracy for VGG19.png')

    plt.figure(2)
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('B1 Training and validation loss for VGG19.png')

def solve_B1_EfficientNet_V2():
    X, y = B1.EfficientNet.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    X_test, y_test = B1.EfficientNet.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    Y_test = B1.EfficientNet.one_hot_encoder(np.array(y_test))
    Y = B1.EfficientNet.one_hot_encoder(np.array(y))
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    model = B1.EfficientNet.create_EfficientNetV2B0_model()
    model.summary()
    BATCH_SIZE = 1000
    SHUFFLE_BUFFER_SIZE = 100
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=40)
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(accuracy))

    plt.figure(1)
    plt.plot(epochs, accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend
    plt.savefig('B1 Training and validation accuracy using EfficientNet-V2.png')

    plt.figure(2)
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('B1 Training and validation loss for EfficientNet-V2.png')

def solve_B2_EfficientNet():
    X, y = B2.EfficientNet.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    X_test, y_test = B2.EfficientNet.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    Y_test = B2.EfficientNet.one_hot_encoder(np.array(y_test))
    Y = B2.EfficientNet.one_hot_encoder(np.array(y))
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    model = B2.EfficientNet.create_EfficientNetV2B0_model()
    model.summary()
    BATCH_SIZE = 1000
    SHUFFLE_BUFFER_SIZE = 100
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=40)
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(accuracy))

    plt.figure(1)
    plt.plot(epochs, accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend
    plt.savefig('B2 Training and validation accuracy before cropping.png')

    plt.figure(2)
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('B2 Training and validation loss before cropping.png')    

def solve_B2_EfficientNet_Cropping():
    X, y = B2.EfficientNet_Cropping.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    X_test, y_test = B2.EfficientNet_Cropping.extract_images_labels(final_image_dir = final_cartoon_image_dir_train, cartoon_images_dir = cartoon_images_dir_train, labels_filename = labels_filename )
    Y_test = B2.EfficientNet_Cropping.one_hot_encoder(np.array(y_test))
    Y = B2.EfficientNet_Cropping.one_hot_encoder(np.array(y))
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    model = B2.EfficientNet_Cropping.create_EfficientNetV2B0_model()
    model.summary()
    BATCH_SIZE = 1000
    SHUFFLE_BUFFER_SIZE = 100
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    history = model.fit(train_dataset, validation_data=(X_valid, Y_valid), epochs=40)
    results = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
    print("test loss, MAE, test acc:", results)
    accuracy = history.history["categorical_accuracy"]
    val_accuracy = history.history["val_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(accuracy))

    plt.figure(1)
    plt.plot(epochs, accuracy, "b", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend
    plt.savefig('B2 Training and validation accuracy after cropping.png')

    plt.figure(2)
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('B2 Training and validation loss after cropping.png')



def main():
    solve_A1_landmark_XGBoost()
    solve_A1_landmark_MLP()
    solve_A1_CNN()
    solve_A2_MLP()
    solve_B1_VGG()
    solve_B1_EfficientNet_V2()
    solve_B2_EfficientNet()
    solve_B2_EfficientNet_Cropping()


if __name__ == "__main__":
    main()