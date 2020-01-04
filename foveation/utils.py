import numpy as np
from os.path import join
from numpy.random import choice


def generate_small_dataset(x, y, n):
    """ We generate the dataset from the (X,y) training tuple,
    using only n example for each class"""
    classes = np.sort(np.unique(y))  # classes
    c = classes.size  # how many

    # we save the index for each class
    list_id_classes = [np.squeeze(np.argwhere(y == y_val_)) for y_val_ in classes]

    new_x = np.zeros((c * n, x.shape[1], x.shape[2]))
    new_y = np.zeros(c * n, dtype=int)
    id_all_classes = np.zeros(c * n, dtype=int)

    for id_class_, ll_ in enumerate(list_id_classes):
        idx_tr = choice(ll_, size=n, replace=False)  # we sample w uniform
        id_all_classes[id_class_*n:(id_class_+1)*n] = idx_tr
        for id_sample_, sample_ in enumerate(idx_tr):
            new_x[n * id_class_ + id_sample_] = x[sample_]
            new_y[n * id_class_ + id_sample_] = y[sample_]

    return new_x, new_y, id_all_classes


def generate_indices(y, n_array, rep, path):
    """ Here we generate the indices for the training set as we require more
    training points and for a fixed amount of repetitions.
    We save the results in a numpy array.
    -------------------
    Parameters:
        y, labels for the training data
        n_array, np.array, each entry is a dataset size
        rep, amount of repetitions
        path, folder where we want to save the results
    -------------------
    Results:
        None
    """
    classes = np.sort(np.unique(y))  # classes
    c = classes.size  # how many

    list_id_classes = [np.squeeze(np.argwhere(y == y_val_)) for y_val_ in classes]  # we save the index for each class

    for n_ in n_array:
        id_all_classes = np.zeros((rep, c * n_), dtype=int)
        for r_ in range(rep):
            for id_class_, ll_ in enumerate(list_id_classes):
                idx_tr = choice(ll_, size=n_, replace=False)  # uniformly
                id_all_classes[r_, id_class_*n_:(id_class_+1)*n_] = idx_tr
        np.save(join(path, 'idx_n_' + str(n_) + '.npy'), id_all_classes)


def generate_index_tr(y, n_tr, rep):
    """ Generate the indices for the smaller training set.
    :parameters:
        y, array of training outputs, ORIGINAL.
        n_tr, number of training samples.
        rep, number of repetitions.
    :returns:
        id_all_classes, tensor of dimensions (rep, n_tr*classes) which contains
        the indices for the examples of the small training set.
    """
    classes = np.sort(np.unique(y))
    c = classes.size
    list_id_classes = [np.squeeze(np.argwhere(y==y_val_))
                       for y_val_ in classes]
    id_all_classes = np.zeros((rep, c*n_tr), dtype=int)
    for r_ in range(rep):
        for id_class_, ll_ in enumerate(list_id_classes):
            idx_tr = choice(ll_, size=n_tr, replace=False)  # uniformly
            id_all_classes[r_, id_class_*n_tr:(id_class_+1)*n_tr] = idx_tr
    return id_all_classes


def one_hot_encoding(y_train, y_test):
    """ We transform the labels using the one-hot encoding.
    :parameter
        y_train, labels for training examples,
        y_test, labels for test examples
    :returns
        y_train_, one hot vector for training
        y_test_, one hot vector for testing
    """
    classes = np.sort(np.unique(y_train))
    n_classes = classes.size
    y_train_ = np.zeros((y_train.size, n_classes), dtype=int)
    y_test_ = np.zeros((y_test.size, n_classes), dtype=int)

    for kk, idx in enumerate(y_train):
        y_train_[kk, idx] = 1
    for kk, idx in enumerate(y_test):
        y_test_[kk, idx] = 1
    return y_train_, y_test_


def generate_index_tr_vl(y, n_tr, rep, n_vl=30000):
    """ y, array of training outputs,
        n_tr, number of training samples,
        rep, number of repetitions
    """
    classes = np.sort(np.unique(y))
    c = classes.size
    list_id_classes = [np.squeeze(np.argwhere(y == y_val_))
                       for y_val_ in classes]
    id_tr = np.zeros((rep, c * n_tr), dtype=int)
    id_vl = np.zeros((rep, c * n_vl), dtype=int)

    for r_ in range(rep):
        for id_class_, ll_ in enumerate(list_id_classes):
            tmp_id_tr = choice(ll_, size=n_tr, replace=False)  # uniformly
            id_tr[r_, id_class_ * n_tr:(id_class_ + 1) * n_tr] = tmp_id_tr

            # int(n_vl/c), validation samples per class
            tmp_id_vl = choice(np.setdiff1d(ll_, tmp_id_tr),  # the order matters
                               size=int(n_vl / c), replace=False)
            id_vl[r_, id_class_ * int(n_vl / c):(id_class_ + 1) * int(n_vl / c)] = tmp_id_vl

    return id_tr, id_vl


