import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from os.path import join
from numpy.random import choice
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, History, TensorBoard, ModelCheckpoint
from utils import generate_indices, generate_small_dataset, plot_history


def noise():
    return

def crop(x, edge=3):
    """ Here we pass the data matrix, the tensor of dimensions (#n_samples, dim1, dim2) """
    _, dim1, dim2 = x.shape
    x_ = x[:, edge:dim1-edge, edge:dim2-edge]
    return x_

def pyramid(x, edge_list=[0, 2, 4, 6, 8, 9, 10]):
    """ Here we make the pyramid, meaning that we zoom into the data """
    n_samples, dim1, dim2 = X.shape
    for i_, e_ in enumerate(edge_list):
        if i_ == 0:
            x_ = np.reshape(crop(x, e_), (n_samples, -1), order='C')
        else:
            x_ = np.hstack((x_, np.reshape(crop(x, e_), (n_samples,-1), order='C')))
    return np.reshape(x_, (n_samples, x_.shape[1], 1))


def main():

    ### change these parameters depending on the experiment
    folder_prediction = 'pyramid_zoomed_in'   # crop , or pyramid
    load_id = True        # if True, retrieve from original
    repetitions = 20
    min_n_train = 1
    max_n_train = 15
    n_epochs = 100000
    tol = 1e-5
    patience = 1
    ########################################################
    folder_results = 'example_results' # results relative to the
    folder_idx = 'indices'
    path_idx = join(folder_results, folder_idx)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print('size: ', x_train.shape)

    # x_train = pyramid(x_train)  # pyramid with all crops
    # x_test = pyramid(x_test)

    x_train = pyramid(x_train, edge_list=[4,6,8])  # pyramid with limited crops
    x_test = pyramid(x_test, edge_list=[4,6,8])
    print('size: ', x_train.shape)

    if load_id: #if we want to retrieve saved training indices
        n_training_array = np.load(join(folder_results, 'n_training_array.npy'))
        list_file = [join(folder_results, folder_idx, 'idx_n_' + str(n_) + '.npy') for n_ in n_training_array]
        print(list_file)
    else: #here we generate new indices
        n_training_array = np.arange(min_n_train, max_n_train)
        np.save(join(folder_results, 'n_training_array.npy'), n_training_array)
        generate_indices(y_train, n_training_array, repetitions, path_idx)

    loss_matrix = np.zeros((repetitions, n_training_array.size))
    acc_matrix = np.zeros((repetitions, n_training_array.size))
    #here we load the dataset
    for id_n_train_, (n_train, file_indices) in enumerate(zip(n_training_array, list_file)):
        indices_across_rep = np.load(file_indices)

        for r_, id_rep in enumerate(indices_across_rep):
            x_train_, y_train_ = x_train[id_rep], y_train[id_rep]

            # if crop, pyramid, noise of something else, modify training data

            _, dim1, dim2 = x_train_.shape
            model = tf.keras.models.Sequential([
              tf.keras.layers.Flatten(input_shape=(dim1, dim2)),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer='SGD',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            #folder_weights_name = 'save_weights'
            #os.makedirs(folder_weights_name, exist_ok=True)
            #checkpoint_path = join(folder_weights_name, 'weights')

            EPOCHS = n_epochs
            csv_logger = CSVLogger(join(folder_results, folder_prediction, 'n_'+str(n_train)+'_r_'+str(r_)+'training.log'))
            #tf_board = TensorBoard()

            early_stop = EarlyStopping(monitor='loss', min_delta=tol, patience=patience)
            #cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

            history = model.fit(x_train_, y_train_, epochs=EPOCHS, validation_split=0, verbose=0, callbacks=[csv_logger, early_stop]) # min_n_train

            loss, acc = model.evaluate(x_test,  y_test)

            loss_matrix[r_, id_n_train_]  = loss
            acc_matrix[r_, id_n_train_] = acc


    np.save(join(folder_results, folder_prediction, 'test_loss.npy'), loss_matrix)
    np.save(join(folder_results, folder_prediction, 'test_acc.npy'), acc_matrix)


if __name__ == '__main__':
    main()
