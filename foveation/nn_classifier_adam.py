import os
import sys
import numpy as np
from os.path import join
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from utils import generate_index_tr_vl


def main():
    """
    :argument
    id_experiment,  # slurm array ID  --> it will correspond to a LR
    exp_paradigm: 1, 2, 3, 4
    data_dim: 28, 36, 40, 56, 80
    """
    n_classes = 10  # mnist classification

    # LEARNING PARADIGM
    # if id_experiment == 1
    id_experiment = sys.argv[1].split('=')[1]
    exp_paradigm = sys.argv[2].split('=')[1] # save in different folders
    # data_dim = sys.argv[3].split('=')[1]  # dimension of the input image

    if id_experiment == str(0):
        data_dim = str(28)
    elif id_experiment == str(1):
        data_dim = str(36)
    elif id_experiment == str(2):
        data_dim = str(40)
    elif id_experiment == str(3):
        data_dim = str(56)
    elif id_experiment == str(4):
        data_dim = str(80)
    else:
        return

    # PATH DEFINITIONS
    path_folder = '/om/user/vanessad/foveation'
    path_code = join(path_folder, 'experiments_nn_model_adam')
    os.makedirs(path_code, exist_ok=True)

    folder_data = join(path_folder, 'modified_MNIST_dataset')
    folder_results = join(path_code, 'results_exp_%s' % exp_paradigm)
    os.makedirs(folder_results, exist_ok=True)

    # EXPERIMENTAL SETUP
    # n_eval_metrics = 4  # (loss, accuracy) on training, (loss, accuracy) on test
    epochs = 100000  # TODO: CONVERGENCE CRITERION!? 0 LOSS?
    repetitions = 1
    # eval_metrics = np.zeros((n_eval_metrics, repetitions))
    n_tr_array = np.arange(1, 15)

    # LOAD DATASET
    mnist = tf.keras.datasets.mnist  # load MNIST dataset
    (_, y_train), (_, y_test) = mnist.load_data()
    x_train = np.load(join(folder_data, 'exp_%s_dim_%s_tr.npy' % (exp_paradigm, data_dim)))
    x_test = np.load(join(folder_data, 'exp_%s_dim_%s_ts.npy' % (exp_paradigm, data_dim)))
    _, dim1, dim2 = x_train.shape

    for n_tr in n_tr_array:
        # this must be done at every repetition of the experiment
        idx_tr_ntr, idx_vl_ntr = generate_index_tr_vl(y_train, n_tr,
                                                      rep=repetitions,
                                                      n_vl=30000)
        if n_tr > 3:
            batchsize = 32
        else:
            batchsize = n_tr * n_classes

        for r_, (idx_tr, idx_vl) in enumerate(zip(idx_tr_ntr, idx_vl_ntr)):
            # MODEL SETUP
            model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(dim1, dim2)),
                                                tf.keras.layers.Dense(128, activation='relu'),
                                                tf.keras.layers.Dense(10, activation='softmax')])
            # TODO: OPTIMIZER
            # TODO: BATCH SIZE
            # TODO: STOPPING CRITERION
            # TODO: CHECK ARCHITECTURE
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-5)

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(x_train[idx_tr], y_train[idx_tr],
                                epochs=epochs,
                                batch_size=batchsize,
                                validation_data=(x_train[idx_vl], y_train[idx_vl]),
                                callbacks=[early_stopping])

            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
            hist = pd.DataFrame(history.history)
            hist.to_csv(join(folder_results, 'history_%s_%s.csv' % (data_dim, str(n_tr))))
            np.save(join(folder_results, 'test_metrics_%s_%s.npy' % (data_dim, str(n_tr))),
                    np.append(test_loss, test_accuracy))


if __name__ == '__main__':
    main()
