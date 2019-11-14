import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from os.path import join
from utils import generate_indices
from tensorflow.keras.callbacks import EarlyStopping


def main():
    """
    BEFORE RUNNING THIS SCRIPT WE NEED TO RUN GENERATE_IMAGES.PY
    We need to specify the type of experiment
        exp_type
            exp_1, exp_2, exp_3, exp_4
        data_dim
            dim_28, 36, 40, 56, 80
        load_idx
            0, generate, 1 load
    """
    id_experiment = sys.argv[1].split('=')[1]
    exp_paradigm = sys.argv[2].split('=')[1]
    data_dim = sys.argv[3].split('=')[1]

    n_array = np.append(np.arange(1, 10), np.append(np.arange(10, 20, 2), np.arange(20, 55, 5)))
    size_n_array = n_array.size
    repetitions = 3

    path_folder = '/om/user/vanessad/foveation'
    path_code = join(path_folder, 'first_exp')

    folder_data = join(path_folder, 'modified_MNIST_dataset')
    folder_indices = join(path_code, 'indices_MNIST_samples_training')
    folder_results = join(path_code, 'results_exp_%s' % exp_paradigm)

    mnist = tf.keras.datasets.mnist  # load mnist dataset
    (_, y_train), (_, y_test) = mnist.load_data()

    del _

    generate_indices(y_train, n_array, repetitions, folder_indices)

    x_train = np.load(join(folder_data, 'exp_%s_dim_%s_tr.npy' % (exp_paradigm, data_dim)))
    x_test = np.load(join(folder_data, 'exp_%s_dim_%s_ts.npy' % (exp_paradigm, data_dim)))
    n_classes = np.unique(y_train).size

    y_train_ = np.zeros((y_train.size, n_classes), dtype=int)
    y_test_ = np.zeros((y_test.size, n_classes), dtype=int)

    for kk, idx in enumerate(y_train):
        y_train_[kk, idx] = 1
    for kk, idx in enumerate(y_test):
        y_test_[kk, idx] = 1

    y_train = y_train_
    y_test = y_test_

    loss_matrix = np.zeros((repetitions, size_n_array))
    acc_matrix = np.zeros((repetitions, size_n_array))

    # to check convergence
    loss_at_train = np.zeros((repetitions, size_n_array))
    acc_at_train = np.zeros((repetitions, size_n_array))

    for id_n_, n_ in enumerate(n_array):
        print('samples: ', n_)
        data_indices_all_rep = np.load(join(folder_indices, 'idx_n_%i.npy' % n_))
        for id_r_, data_id_rep_ in enumerate(data_indices_all_rep):

            x_train_, y_train_ = x_train[data_id_rep_], y_train[data_id_rep_]
            _, dim1, dim2 = x_train_.shape
            model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(dim1, dim2)),
                    tf.keras.layers.Dense(10, activation='linear')])

            model.compile(optimizer='SGD',  # nothing should change -- uniqueness
                          loss='mean_squared_error',
                          metrics=['accuracy'])

            # csv_logger = CSVLogger(join(path_experiment, 'n_%s_r_%s_training.log' % (id_n_, id_r_)))
            overfit_stop = EarlyStopping(monitor='loss', min_delta=1e-5, patience=20)

            history = model.fit(x_train_, y_train_, epochs=100000,
                                validation_split=0, verbose=0,
                                callbacks=[overfit_stop])  # min_n_train

            hist = pd.DataFrame(history.history)
            acc_at_train[id_r_, id_n_] = hist['accuracy'].values[-1]
            loss_at_train[id_r_, id_n_] = hist['loss'].values[-1]
            if hist['accuracy'].values[-1] != 1:
                return

            loss, acc = model.evaluate(x_test, y_test)
            loss_matrix[id_r_, id_n_] = loss
            acc_matrix[id_r_, id_n_] = acc

    del x_train, x_test, x_train_, y_train, y_test

    metrics_all = np.zeros((4, repetitions, size_n_array))
    metrics_all[0] = loss_at_train
    metrics_all[1] = acc_at_train
    metrics_all[2] = loss_matrix
    metrics_all[3] = acc_matrix

    np.save(join(folder_results, 'metrics_%s_%s.npy' % (data_dim, id_experiment)), metrics_all)


if __name__ == '__main__':
    main()
