import numpy as np
from numpy.random import choice


def _generate_index_tr_vl(y, n_tr, n_vl=30000):
    """ We generate here the indices for the training and validation sets.
    WARNING: ORIGINAL Ys (no hot encoding)
    :param y: array containing the different classes from the training set.
    :param n_tr: number of training samples
    :param n_vl: total number of validation samples
    :return: indices
    """
    classes = np.sort(np.unique(y))  # which are the main categories
    c = classes.size  # number of classes
    list_id_classes = [np.squeeze(np.argwhere(y == y_val_))
                       for y_val_ in classes]  # we divide the indices for the different categories

    n_vl = int(n_vl / c)  # number of validation samples per class

    id_tr = np.zeros(c * n_tr, dtype=int)  # indices of training
    id_vl = np.zeros(c * n_vl, dtype=int)  # indices of validation

    for id_class_, ll_ in enumerate(list_id_classes):  # for each category
        tmp_id_tr = choice(ll_, size=n_tr, replace=False)  # uniformly sampled
        id_tr[id_class_ * n_tr:(id_class_ + 1) * n_tr] = tmp_id_tr

        tmp_id_vl = choice(np.setdiff1d(ll_, tmp_id_tr),  # the order matters
                           size=n_vl, replace=False)
        id_vl[id_class_ * n_vl:(id_class_ + 1) * n_vl] = tmp_id_vl

    return id_tr, id_vl
