#  linear classifier using gradient descent

import sys
import numpy as np
from numpy.random import choice
import tensorflow as tf
import pandas as pd
from os.path import join
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping


def main():
    """
    BEFORE RUNNING THIS SCRIPT WE NEED TO RUN GENERATE_IMAGES.PY
    :argument
        id_experiment, # slurm
        exp_paradigm:  1, 2, 3, 4
        data_dim: 28, 36, 40, 56, 80
        lr_val: float value, learning rate
    """
    # LEARNING PARADIGM
    id_experiment = sys.argv[1].split('=')[1]
    exp_paradigm = sys.argv[2].split('=')[1]
    data_dim = sys.argv[3].split('=')[1]
    lr_val = float(sys.argv[4].split('=')[1])
    lr_str = str(np.format_float_scientific(lr_val, precision=0, unique=True, trim='-'))

    # PATH DEFINITIONS
    path_folder = '/om/user/vanessad/foveation'
    path_code = join(path_folder, 'experiments_linear_model')
    folder_data = join(path_folder, 'modified_MNIST_dataset')
    folder_results = join(path_code, 'results_exp_%s' % exp_paradigm)
    print('folder_results', folder_results)

    # EXPERIMENTAL SETUP
    n_array = np.append(np.arange(1, 10),
                        np.append(np.arange(10, 20, 2),
                                  np.arange(20, 150, 10)))
    # n_array = np.array([10, 20])  # TESTING
    size_n_array = n_array.size
    repetitions = 3
    n_eval_metrics = 4  # (loss, accuracy) on training, (loss, accuracy) on test
    epochs = 100000

    # LOAD DATASET
    mnist = tf.keras.datasets.mnist  # load MNIST dataset
    x_train = np.load(join(folder_data, 'exp_%s_dim_%s_tr.npy' % (exp_paradigm, data_dim)))
    x_test = np.load(join(folder_data, 'exp_%s_dim_%s_ts.npy' % (exp_paradigm, data_dim)))
    (_, y_train), (_, y_test) = mnist.load_data()
    y_train_hot_enc, y_test_hot_enc = one_hot_encoding(y_train, y_test)

    eval_metrics = np.zeros((n_eval_metrics, repetitions, size_n_array))

    # TRAINING AND EVALUATION
    for id_n_, n_ in enumerate(n_array):
        indices_w_fixed_n = generate_index_tr(y_train, n_, repetitions)

        for id_r_, data_id_rep_ in enumerate(indices_w_fixed_n):
            x_train_ = x_train[data_id_rep_]
            y_train_ = y_train_hot_enc[data_id_rep_]
            _, dim1, dim2 = x_train_.shape

            # MODEL SETUP
            model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(dim1, dim2)),
                    tf.keras.layers.Dense(10, activation='linear')])
            sgd = SGD(lr=lr_val, decay=1e-6,  momentum=0., nesterov=False)
            model.compile(optimizer=sgd,
                          loss='mean_squared_error',
                          metrics=['accuracy'])
            overfit_stop = EarlyStopping(monitor='loss', min_delta=1e-5, patience=20)
            # TODO: CHANGE
            #  we need some validation set, but in the linear regime we want overfitting

            if n_ < 3:
                batch_size = None
            else:
                batch_size = 32
            history = model.fit(x_train_, y_train_,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0,
                                validation_data=None,
                                verbose=0,
                                callbacks=[overfit_stop])  # min_n_train
            # TODO: CHANGE increase the number of epochs to 100k
            hist = pd.DataFrame(history.history)  # logs from training

            loss_test, acc_test = model.evaluate(x_test, y_test_hot_enc)  # evaluation of test

            eval_metrics[0, id_r_, id_n_] = hist['loss'].values[-1]
            eval_metrics[1, id_r_, id_n_] = hist['accuracy'].values[-1]
            eval_metrics[2, id_r_, id_n_] = loss_test
            eval_metrics[3, id_r_, id_n_] = acc_test

    np.save(join(folder_results, 'metrics_%s_%s_%s.npy' % (lr_str, data_dim, id_experiment)),
            eval_metrics)


if __name__ == '__main__':
    main()
