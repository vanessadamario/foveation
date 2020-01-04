import os
import sys
import numpy as np
from os.path import join
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import generate_index_tr_vl


def main():
    """
    :argument
    id_experiment,  # slurm array ID  --> it will correspond to a LR
    exp_paradigm: 1, 2, 3, 4
    data_dim: 28, 36, 40, 56, 80
    """
    lr_array = np.array([0.001, 0.005, 0.01, 0.02, 0.03,
                         0.08, 0.1, 0.2, 0.3, 0.5])
    # LEARNING PARADIGM
    id_experiment = int(sys.argv[1].split('=')[1])
    lr_val = lr_array[id_experiment]
    lr_str = str(np.format_float_scientific(lr_val, precision=0, unique=True, trim='-'))

    exp_paradigm = sys.argv[2].split('=')[1]  # save in different folders

    data_dim = sys.argv[3].split('=')[1]

    # PATH DEFINITIONS
    path_folder = '/om/user/vanessad/foveation'
    path_code = join(path_folder, 'experiments_nn_model')
    os.makedirs(path_code, exist_ok=True)

    folder_data = join(path_folder, 'modified_MNIST_dataset')
    folder_results = join(path_code, 'results_exp_%s' % exp_paradigm)
    os.makedirs(folder_results, exist_ok=True)
    # TODO: COMMENT THE FOLLOWING LINES
    # folder_results = join(path_code, 'results_example')
    # os.makedirs(folder_results, exist_ok=True)

    # EXPERIMENTAL SETUP
    # n_eval_metrics = 4  # (loss, accuracy) on training, (loss, accuracy) on test
    epochs = 10000  # 100000
    repetitions = 1
    # eval_metrics = np.zeros((n_eval_metrics, repetitions))
    n_tr = 10

    # LOAD DATASET
    mnist = tf.keras.datasets.mnist  # load MNIST dataset
    (_, y_train), (_, y_test) = mnist.load_data()
    x_train = np.load(join(folder_data, 'exp_%s_dim_%s_tr.npy' % (exp_paradigm, data_dim)))
    x_test = np.load(join(folder_data, 'exp_%s_dim_%s_ts.npy' % (exp_paradigm, data_dim)))
    _, dim1, dim2 = x_train.shape

    # this must be done at every repetition of the experiment
    idx_tr, idx_vl = generate_index_tr_vl(y_train, n_tr,
                                          rep=repetitions,
                                          n_vl=30000)
    idx_tr = idx_tr[0]  # TODO: remove -- evaluate over all repetitions
    idx_vl = idx_vl[0]
    # MODEL SETUP
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(dim1, dim2)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # TODO: OPTIMIZER
    # TODO: BATCH SIZE
    # TODO: STOPPING CRITERION
    # TODO: CHECK ARCHITECTURE
    sgd = SGD(lr=lr_val, decay=1e-6,  momentum=0., nesterov=False)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                     patience=5, min_lr=0.001)
    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train[idx_tr], y_train[idx_tr],
                        epochs=epochs,
                        validation_data=(x_train[idx_vl], y_train[idx_vl]),
                        callbacks=[lr_reduction])

    model.evaluate(x_test, y_test, verbose=2)
    hist = pd.DataFrame(history.history)
    hist.to_csv(join(folder_results, 'history_%s_$s.csv' % (data_dim, lr_str)))


if __name__ == '__main__':
    main()