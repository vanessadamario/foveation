import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import choice
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, History, TensorBoard, ModelCheckpoint


def generate_small_dataset(x, y, n):
    """ We generate the dataset from the (X,y) training tuple, using only n example for each class"""
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

    # for (x_tr_, y_tr_) in zip(new_X, new_y):  # check if it works
    #    plt.imshow(x_tr_, cmap='gray')
    #    plt.title(str(y_tr_))
    #    plt.show()
    #     plt.close()
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

    list_id_classes = [np.squeeze(np.argwhere(y == y_val_)) for y_val_ in classes] # we save the index for each class

    for n_ in n_array:
        id_all_classes = np.zeros((rep, c * n_), dtype=int)
        for r_ in range(rep):
            for id_class_, ll_ in enumerate(list_id_classes):
                idx_tr = choice(ll_, size=n_, replace=False)  # uniformly
                id_all_classes[r_, id_class_*n_:(id_class_+1)*n_] = idx_tr
        np.save(join(path, 'idx_n_' + str(n_) + '.npy'), id_all_classes)


"""
def plot_history(history, valid=False):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Sparse Categorical Crossentropy')
    plt.plot(hist['loss'], label='Train Error')
    if valid:
        plt.plot(hist['val_loss'], label = 'Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['accuracy'], label='Train Accuracy')
    if valid:
        plt.plot(hist['val_accuracy'], label = 'Val Error')
    plt.legend()
    plt.show()
"""