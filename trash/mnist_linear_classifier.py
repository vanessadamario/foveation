import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
from sklearn.preprocessing import OneHotEncoder
from foveation.utils import generate_indices
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

N_ARRAY = np.arange(1, 30)
REPETITIONS = 2
EDGE_LIST = [0, 2, 4, 6, 8, 9, 10]


def crop(x, edge=3):
    """ Here we generate a single crop for the image,
    by specifying the number of pixels we want to get rid from, for each side.
    We pass as input the data matrix, the tensor of dimensions (#n_samples, dim1, dim2).
    -----------------
    Parameters:
        x, input dataset, tensor of dimensions (#n_samples, dim1, dim2),
        edge, number of pixels we want to remove from each side
    -----------------
    Returns:
        x_, cropped version of the original image, dimensions (#n_samples, dim1-2*edge, dim2-2*edge)
     """
    _, dim1, dim2 = x.shape
    x_ = x[:, edge:dim1-edge, edge:dim2-edge]
    return x_


def increase_redundant_dataset_dimensions(x, edge_list=EDGE_LIST):
    """ Here we generate the higher dimensional data by taking as input the output of crops.
    -----------------
    Parameters:
        x, input dataset, tensor of dimensions (#n_samples, dim1, dim2),
        edge_list, list of number of pixels we want to remove from each side
    -----------------
    Returns:
        x_, high dimensional array containing the original dataset, plus the cropped
         version of the image(#n_samples, dim1-2*edge, dim2-2*edge)
     """
    n_samples, dim1, dim2 = x.shape
    for i_, e_ in enumerate(edge_list):
        if i_ == 0:
            x_ = np.reshape(crop(x, e_), (n_samples, -1), order='C')
        else:
            x_ = np.hstack((x_, np.reshape(crop(x, e_), (n_samples, -1), order='C')))
    return np.reshape(x_, (n_samples, x_.shape[1], 1))


def increase_noisy_dataset_dimensions(x, edge_list=EDGE_LIST, noise_var=1e-1):
    """ Here we generate the higher dimensional dataset, by appending random noise, additional features.
    -----------------
    Parameters:
        x, input dataset, tensor of dimensions (#n_samples, dim1, dim2),
        edge_list, list of number of pixels we want to remove from each side
    -----------------
    Returns:
        x_, high dimensional array containing the original dataset, plus pure noise,
         as in increase_redundant_dataset_dimensions
     """
    n_samples, dim1, dim2 = x.shape

    for i_, e_ in enumerate(edge_list):
        if i_ == 0:
            x_ = np.reshape(crop(x, e_), (n_samples, -1), order='C')
        else:
            x_ = np.hstack((x_, noise_var * np.random.randn(n_samples, (dim1-(2*e_))*(dim1-(2*e_)))))

    return np.reshape(x_, (n_samples, x_.shape[1], 1))


def increase_mixed_dataset_dimensionality(x, edge_list=EDGE_LIST):
    """ Here we generate the higher dimensional dataset, by appending to the original image,
    some noisy zooms
    -----------------
    Parameters:
        x, input dataset, tensor of dimensions (#n_samples, dim1, dim2),
        edge_list, list of number of pixels we want to remove from each side
    -----------------
    Returns:
        x_, high dimensional array containing the original dataset, plus noisy zoomed versions of the image,
         as in increase_redundant_dataset_dimensions
     """
    n_samples, dim1, dim2 = x.shape

    for i_, e_ in enumerate(edge_list):
        if i_ == 0:
            x_ = np.reshape(crop(x, e_), (n_samples, -1), order='C')
        else:
            x_ = np.hstack((x_, (np.reshape(crop(x, e_), (n_samples, -1), order='C') +
                                 1e-1 * np.random.randn((dim1-(2*e_))*(dim1-(2*e_))))))

    return np.reshape(x_, (n_samples, x_.shape[1], 1))


def generate_noisy_images(x, noise_var=1e-1):
    """ Here we generate the noisy version of the image. We put a random noise generated using a Gaussian distribution.
     The mean is centered at 0, the variance is equivalent to noise_var.
     We preserve the original dimension of the image, even though the learning process gets harder, we have
     lesser 0 values.
     -----------------
    Parameters:
        x, input dataset, tensor of dimensions (#n_samples, dim1, dim2),
        noise_var, noise level
     -----------------
    Returns:
        noisy_x, the noisy version of the dataset, where we preserve the original data
     """
    n, dim1, dim2 = x.shape
    noisy_x = noise_var * np.random.randn(n, dim1, dim2)
    noisy_x[x != 0] = x[x != 0]
    return noisy_x


# ask Xavier if Early stopping with tol on the training loss makes sense
DICT_PARADIGM = {'mixed': increase_mixed_dataset_dimensionality,
                 'noisy': increase_noisy_dataset_dimensions,
                 'red': increase_redundant_dataset_dimensions}

KERAS_LOSSES = {'square': 'mean_squared_error',
                'cross': 'categorical_crossentropy'}
EPOCHS = 10000


def main():
    """ We need to pass as input several parameters
        sys.argv[1], crops, this implies number of additional features
        sys.argv[2], bool flag, if 1 we load the previous indices,
        sys.argv[3], str, experimental paradigm: 'mixed', 'red', 'noisy',
        sys.argv[4], str, loss type: 'square', 'cross'
    """

    NOISY_FLAG = 1

    max_zoom = int(sys.argv[1])  # number of scales we want to give
    if max_zoom < 1 or max_zoom > len(EDGE_LIST):
        raise ValueError('max_zoom must be an int in [1, len(EDGE_LIST)-1]')
    flag_load, exp_type, loss_type = sys.argv[-3:]
    flag_load = bool(int(flag_load))
    size_n_array = N_ARRAY.size

    common_path = './results_all'
    path_experiment = '%s_%s' % (exp_type, loss_type)
    os.makedirs(common_path, exist_ok=True)  # we generate the path where indices common to all experiments
    os.makedirs(path_experiment, exist_ok=True)  # we generate the path with the experiments

    mnist = tf.keras.datasets.mnist # load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    _, dim1, dim2 = x_test.shape

    if NOISY_FLAG:
        x_train = generate_noisy_images(x_train)
        x_test = generate_noisy_images(x_test)

    feature_set_size = np.sum(np.array([(dim1 - 2*ee) * (dim2 - 2*ee)
                                        for ee in EDGE_LIST[:max_zoom]]))

    x_train = DICT_PARADIGM[exp_type](x_train)
    x_test = DICT_PARADIGM[exp_type](x_test)

    x_train = x_train[:, :feature_set_size]  # number of additional features
    x_test = x_test[:, :feature_set_size]

    # generic check on the indices
    if not flag_load:  # generate new indices
        generate_indices(y_train, N_ARRAY, REPETITIONS, path=common_path)
    elif flag_load and not np.all(np.array(['idx_n_%i.npy' % n_ in os.listdir(common_path)
                                            for n_ in N_ARRAY])):
        raise ValueError('Mismatch in indices and n_array')

    # np.save(join(path_experiment, 'x_train.npy'), x_train)
    np.save(join(path_experiment, 'x_test.npy'), x_test)

    enc = OneHotEncoder(categories='auto')
    enc.fit((np.unique(y_train)).reshape(-1, 1))
    y_train = (enc.transform(y_train.reshape(-1, 1))).toarray()
    y_test = (enc.transform(y_test.reshape(-1, 1))).toarray()

    loss_matrix = np.zeros((REPETITIONS, size_n_array))
    acc_matrix = np.zeros((REPETITIONS, size_n_array))

    for id_n_, n_ in enumerate(N_ARRAY):
        print('samples: ', n_)
        data_indices_all_rep = np.load(join(common_path, 'idx_n_%i.npy' % n_))
        for id_r_, data_id_rep_ in enumerate(data_indices_all_rep):

            x_train_, y_train_ = x_train[data_id_rep_], y_train[data_id_rep_]
            _, dim1, dim2 = x_train_.shape
            model = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape=(dim1, dim2)),
                    tf.keras.layers.Dense(10, activation='linear')])

            # sgd = SGD(learning_rate=5e-6, momentum=0., nesterov=False, clipnorm=1.)
            model.compile(optimizer='SGD',  # nothing should change -- uniqueness
                          loss=KERAS_LOSSES[loss_type],
                          metrics=['accuracy'])

            print(model.summary())

            csv_logger = CSVLogger(join(path_experiment, 'n_%s_r_%s_training.log' % (id_n_, id_r_)))
            overfit_stop = EarlyStopping(monitor='loss', min_delta=1e-5, patience=20)

            history = model.fit(x_train_, y_train_, epochs=EPOCHS,
                                validation_split=0, verbose=0,
                                callbacks=[csv_logger, overfit_stop])  # min_n_train

            hist = pd.DataFrame(history.history)

            if hist['accuracy'].values[-1] != 1:
                return

            loss, acc = model.evaluate(x_test, y_test)
            loss_matrix[id_r_, id_n_] = loss
            acc_matrix[id_r_, id_n_] = acc


if __name__ == '__main__':
    main()
