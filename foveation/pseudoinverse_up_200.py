import os
import sys
import numpy as np
import tensorflow as tf
from numpy.random import choice
from os.path import join
from numpy.linalg import matrix_rank, svd
from utils import generate_indices


def compute_accuracy(y_true, y_pred):
    """ y_pred is here the outcome of the square loss function
    :param: y_true, the array with one hot encoding
    :param: y_pred, one hot encoding, from the regression problem
    :return: accuracy, scalar value """
    n_samples, n_classes = y_true.shape
    y_pred_sparse = np.zeros_like(y_true)
    for i_, pos in enumerate(np.argmax(y_pred, axis=-1)):
        y_pred_sparse[i_, pos] = 1
    discrepancy = np.abs(y_true - y_pred_sparse).sum(axis=-1)
    well_classified = float(np.where(discrepancy==0)[0].size)
    return well_classified / n_samples


def generate_indices(y, n_, rep):
    """ Here we generate the indices for the training set as we require more
    training points and for a fixed amount of repetitions.
    We save the results in a numpy array.
    -------------------
    Parameters:
        y, labels for the training data
        n_, scalar, training dataset size
        path, folder where we want to save the results
    -------------------
    Results:
        None
    """
    classes = np.sort(np.unique(y))  # classes
    c = classes.size  # how many

    list_id_classes = [np.squeeze(np.argwhere(y == y_val_))
                       for y_val_ in classes]  # we save the index for each class

    id_all_classes = np.zeros((rep, c*n_), dtype=int)
    for r_ in range(rep):
        for id_class_, ll_ in enumerate(list_id_classes):
            idx_tr = choice(ll_, size=n_, replace=False)  # uniformly
            id_all_classes[r_, id_class_*n_:(id_class_+1)*n_] = idx_tr
    return id_all_classes


def main():

    exp_paradigm = sys.argv[1].split('=')[1]
    data_dim = sys.argv[2].split('=')[1]

    n_array = np.append(np.arange(1, 10),
                        np.append(np.arange(10, 20, 2),
                        np.arange(20, 150, 10)))
    size_n_array = n_array.size
    repetitions = 30

    path_folder = '/om/user/vanessad/foveation'
    path_code = join(path_folder, 'pseudoinverse_up_200')
    os.makedirs(path_code, exist_ok=True)

    folder_data = join(path_folder, 'modified_MNIST_dataset')
    folder_results = join(path_code, 'results_exp_%s' % exp_paradigm)
    os.makedirs(folder_results, exist_ok=True)

    mnist = tf.keras.datasets.mnist  # load mnist dataset
    (_, y_train), (_, y_test) = mnist.load_data()
    y_train_no_one_hot = y_train.copy()

    del _

    x_train = np.load(join(folder_data, 'exp_%s_dim_%s_tr.npy' % (exp_paradigm, data_dim)))
    x_test = np.load(join(folder_data, 'exp_%s_dim_%s_ts.npy' % (exp_paradigm, data_dim)))
    n_ts, _, _ = x_test.shape
    x_test = x_test.reshape(n_ts, -1)

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
    rank_data_matrix = np.zeros((repetitions, size_n_array))
    min_eigen = np.zeros((repetitions, size_n_array))

    for id_n_, n_ in enumerate(n_array):
        data_indices_all_rep = generate_indices(y_train_no_one_hot, n_, repetitions)

        for id_r_, data_id_rep_ in enumerate(data_indices_all_rep):
            x_train_, y_train_ = x_train[data_id_rep_], y_train[data_id_rep_]
            x_train_ = x_train_.reshape(n_ * n_classes, -1)
            samples_, features_ = x_train_.shape

            if id_r_ == 0:
                print('samples: ', samples_)
                print('features: ', features_)

            if samples_ < features_:
                try:
                    prod = np.dot(x_train_, x_train_.T)
                    print('rank of product: ', matrix_rank(prod))
                    gram = np.linalg.inv(prod)  # this has dimensions n \times n
                    sol_pinv = np.dot(x_train_.T, gram).dot(y_train_)
                    rank_data_matrix[id_r_, id_n_] = matrix_rank(gram)
                    min_eigen[id_r_, id_n_] = np.min(svd(gram, compute_uv=False))
                    y_pred = np.dot(x_test, sol_pinv)
                except:
                    y_pred = np.zeros_like(y_test)
                    print('Not invertible - in n times n')

            else:
                try:
                    prod = np.dot(x_train_.T, x_train_)
                    print('rank of product: ', matrix_rank(prod))
                    inv_closed_form = np.linalg.inv(prod)  # this has dimensions p \times p
                    sol_pinv = np.dot(inv_closed_form, np.dot(x_train_.T, y_train_))
                    rank_data_matrix[id_r_, id_n_] = matrix_rank(inv_closed_form)
                    min_eigen[id_r_, id_n_] = np.min(svd(inv_closed_form, compute_uv=False))
                    y_pred = np.dot(x_test, sol_pinv)
                except:
                    y_pred = np.zeros_like(y_test)
                    print('Not invertible - in p times p')


            loss_matrix[id_r_, id_n_] = (1./n_ts)*np.sum((y_pred - y_test)**2)
            acc_matrix[id_r_, id_n_] = compute_accuracy(y_test, y_pred)

    del x_train, x_test, x_train_, y_train, y_test

    metrics_all = np.zeros((2, repetitions, size_n_array))
    metrics_all[0] = loss_matrix
    metrics_all[1] = acc_matrix

    np.save(join(folder_results, 'min_eigenvalue_%s.npy' % (data_dim)),
            min_eigen)
    np.save(join(folder_results, 'matrix_rank_%s.npy' % (data_dim)),
            rank_data_matrix)
    np.save(join(folder_results, 'metrics_pinv_%s.npy' % (data_dim)),
            metrics_all)


if __name__ == '__main__':
    main()
