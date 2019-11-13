import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
import matplotlib.pyplot as plt


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


def augment_size(x, edge):
    """
    We aument the original size of each image, by putting it in the center
    :param x: original image, tensor of size n_samples, dim_x, dim_y
    :param edge: number of pixels we want to enlarge the image for each size
    :return out: enlarged tensor
    """
    n_samples, dim1, dim2 = x.shape
    out = np.zeros((n_samples, 2*edge+dim1, 2*edge+dim2))
    out[:, edge:edge+dim1, edge:edge+dim2] = x
    return out


def augment_size_add_noise(x, edge, noise_var=2e-1):
    """
    Add noise to the original image.
    :param x: tensor of size n_samples, dim_x, dim_y
    :param edge: number of pixels we want to enlarge the image for each size of the image
    :param noise_var: noise variance, Gaussian noise
    :return:
    """
    n_samples, dim1, dim2 = x.shape
    out = noise_var * np.abs(np.random.randn(n_samples, 2*edge+dim1, 2*edge+dim2))
    out[:, edge:edge+dim1, edge:edge+dim2] = x
    return out


def upscale(x, e=2):
    """ We generate here the up-sampled version of an image."""
    n_samples, dim1, dim2 = x.shape  # original data
    c = 0
    ll = np.zeros(dim1, dtype=int)

    for k in range(dim1):
        ll[k] = np.round(c).astype(int)
        c += (dim1 + 2 * e) / dim1  # we generate the indices for the new image

    id_last_position_possible = dim1 + 2 * e - 1  # last acceptable value for the mask

    # we would start filling the sparse largest matrix from index (0,0), non symmetrical to the center
    discrepancy = id_last_position_possible - ll[-1]  # we take into account for asymmetries here
    ll += (discrepancy // 2)
    min_diff = np.min(np.diff(ll))  # the minimum gap - we average on tiles of this dimension

    x_ = np.zeros((n_samples, dim1 + 2 * e, dim2 + 2 * e))
    mesh_y, mesh_x = np.meshgrid(ll, ll)  # mesh-grid generation
    x_[:, mesh_x, mesh_y] = x  # up-sampling, original pixels, with lots of holes
    tmp_x = np.zeros((n_samples, dim1 + 2 * e + 2 * min_diff, dim2 + 2 * e + 2*min_diff))
    tmp_x[:, min_diff:dim1 + 2 * e + min_diff,  min_diff: dim2 + 2 * e + min_diff] = x_

    x__ = np.zeros((n_samples, dim1 + 2 * e + 2 * min_diff, dim2 + 2 * e + 2 * min_diff))

    # we fill the holes from the up-sampled version here
    for n in range(n_samples):
        for i in range(min_diff, dim1 + 2 * e + min_diff):
            for j in range(min_diff, dim2 + 2 * e + min_diff):
                tmp_array = np.array([tmp_x[n, i - min_diff:i + min_diff + 1,
                                            j - min_diff:j + min_diff + 1]])  # averaging on tiles
                tmp_non_zeros = np.count_nonzero(tmp_array)

                if tmp_non_zeros > 0:
                    x__[n, i, j] = np.sum(tmp_array) / tmp_non_zeros  #  normalization

    return x__[:, min_diff: dim1 + 2 * e + min_diff, min_diff: dim2 + 2 * e + min_diff]


"""
def upscale(x, e=2):
    n_samples, dim1, dim2 = x.shape  # original data
    c = 0
    ll = np.zeros(dim1, dtype=int)

    for k in range(dim1):
        ll[k] = np.round(c).astype(int)
        c += (dim1 + 2 * e) / dim1  # we generate the indices for the new image
    min_diff = np.min(np.diff(ll))

    x_ = np.zeros((n_samples, dim1 + 2 * e, dim2 + 2 * e))
    mesh_y, mesh_x = np.meshgrid(ll, ll)
    x_[:, mesh_x, mesh_y] = x  # up-sampling, original pixels, with lots of holes

    x__ = # x_.copy()

    # we fill the holes from the up-sampled version here
    for n in range(n_samples):
        for i in range(min_diff, dim1 + 2 * e - min_diff):
            for j in range(min_diff, dim2 + 2 * e - min_diff):
                tmp_array = np.array([x_[n, i - min_diff:i + min_diff + 1,
                                      j - min_diff:j + min_diff + 1]])
                tmp_non_zeros = np.count_nonzero(tmp_array)
                if tmp_non_zeros > 0:
                    x__[n, i, j] = np.sum(tmp_array) / tmp_non_zeros  #  normalization

    return x__
"""

def main():

    mnist = tf.keras.datasets.mnist  # load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    _, dim1, dim2 = x_test.shape

    edge_list = [4, 6, 14, 26]
    x_train_small = x_train[:10]
    folder_plots = 'new_template_images'
    os.makedirs(folder_plots, exist_ok=True)

    # experiment 1
    print('exp 1')
    exp_1_list = []
    for i_, e_ in enumerate(edge_list):
        x_train_ = augment_size_add_noise(x_train_small, e_)
        print(x_train_.shape)
        exp_1_list.append(x_train_)
        # plt.imshow(x_train_[0])
        # plt.colorbar()
        # plt.savefig(join(folder_plots, 'exp_1_%i.pdf' % i_))
        # plt.show()
        # plt.close()

    print('exp 2')
    exp_2_list = []
    for i_, e_ in enumerate(edge_list):
        x_train_ = upscale(x_train_small, e_)
        print(x_train_.shape)
        exp_2_list.append(x_train_)
        plt.imshow(x_train_[0])
        plt.colorbar()
        plt.savefig(join(folder_plots, 'exp_2_%i.pdf' % i_))
        plt.show()
        plt.close()

    print('exp 3, 4')
    for i_, (e_, exp_1_, exp_2_) in enumerate(zip(edge_list, exp_1_list, exp_2_list)):
        _, d1, d2 = exp_1_.shape
        print((150 - d1) / 2)
        up_factor = int((150 - d1) / 2)
        x_train_1 = upscale(exp_1_, up_factor)
        print(x_train_1.shape)
        plt.imshow(x_train_1[0])
        plt.colorbar()
        plt.savefig(join(folder_plots, 'exp_3_%i.pdf' % i_))
        plt.show()
        plt.close()

        x_train_2 = augment_size_add_noise(exp_2_, up_factor)
        print(x_train_2.shape)
        plt.imshow(x_train_2[0])
        plt.colorbar()
        plt.savefig(join(folder_plots, 'exp_4_%i.pdf' % i_))
        plt.show()
        plt.close()

    return


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
