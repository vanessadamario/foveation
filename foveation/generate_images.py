import os
import numpy as np
import tensorflow as tf
from os.path import join


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
    """ We aument the original size of each image, by putting it in the center
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


def main():

    mnist = tf.keras.datasets.mnist  # load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    _, dim1, dim2 = x_test.shape

    edge_list = [0, 4, 6, 14, 26]
    x_train = x_train
    x_test = x_test

    folder_dataset = '/om/user/vanessad/foveation/modified_MNIST_dataset'

    os.makedirs(folder_dataset, exist_ok=True)

    # experiment 1
    print('exp 1')
    for i_, e_ in enumerate(edge_list):
        x_train_ = augment_size_add_noise(x_train, e_)
        _, dim_x, dim_y = x_train_.shape
        np.save(join(folder_dataset, 'exp_1_dim_%i_tr.npy' % dim_x), x_train_)
    for i_, e_ in enumerate(edge_list):
        x_test_ = augment_size_add_noise(x_test, e_)
        _, dim_x, dim_y = x_test_.shape
        np.save(join(folder_dataset, 'exp_1_dim_%i_ts.npy' % dim_x), x_test_)

    print('exp 2')
    for i_, e_ in enumerate(edge_list):
        x_train_ = upscale(x_train, e_)
        _, dim_x, dim_y = x_train_.shape
        np.save(join(folder_dataset, 'exp_2_dim_%i_tr.npy' % dim_x), x_train_)
    for i_, e_ in enumerate(edge_list):
        x_test_ = upscale(x_test, e_)
        _, dim_x, dim_y = x_test_.shape
        np.save(join(folder_dataset, 'exp_2_dim_%i_ts.npy' % dim_x), x_test_)
    del x_train_, x_test_

    print('exp 3')
    list_filename_exp_1_tr = ['exp_1_dim_%i_tr.npy' % (2 * e_ + 28) for e_ in edge_list]
    for i_, (e_, filename_exp_1_tr) in enumerate(zip(edge_list, list_filename_exp_1_tr)):
        exp_1_ = np.load(join(folder_dataset, filename_exp_1_tr))
        _, d1, d2 = exp_1_.shape
        up_factor = int((150 - d1) / 2)
        x_train_3 = upscale(exp_1_, up_factor)
        np.save(join(folder_dataset, 'exp_3_dim_%i_tr.npy' % d1), x_train_3)
    del x_train_3

    list_filename_exp_1_ts = ['exp_1_dim_%i_ts.npy' % (2 * e_ + 28) for e_ in edge_list]
    for i_, (e_, filename_exp_1_ts) in enumerate(zip(edge_list, list_filename_exp_1_ts)):
        exp_1_ = np.load(join(folder_dataset, filename_exp_1_ts))
        _, d1, d2 = exp_1_.shape
        up_factor = int((150 - d1) / 2)
        x_test_3 = upscale(exp_1_, up_factor)
        np.save(join(folder_dataset, 'exp_3_dim_%i_ts.npy' % d1, x_test_3))
    del x_test_3

    print('exp 4')
    list_filename_exp_2_tr = ['exp_2_dim_%i_tr.npy' % (2 * e_ + 28) for e_ in edge_list]
    for i_, (e_, filename_exp_2_tr) in enumerate(zip(edge_list, list_filename_exp_2_tr)):
        exp_2_ = np.load(join(folder_dataset, filename_exp_2_tr))
        _, d1, d2 = exp_2_.shape
        up_factor = int((150 - d1) / 2)
        x_train_4 = augment_size_add_noise(exp_2_, up_factor)
        np.save(join(folder_dataset, 'exp_4_dim_%i_tr.npy' % d1), x_train_4)
    del x_train_4

    list_filename_exp_2_ts = ['exp_2_dim_%i_ts.npy' % (2 * e_ + 28) for e_ in edge_list]
    for i_, (e_, filename_exp_2_ts) in enumerate(zip(edge_list, list_filename_exp_2_ts)):
        exp_2_ = np.load(join(folder_dataset, filename_exp_2_ts))
        _, d1, d2 = exp_2_.shape
        up_factor = int((150 - d1) / 2)
        x_test_4 = augment_size_add_noise(exp_2_, up_factor)
        np.save(join(folder_dataset, 'exp_4_dim_%i_ts.npy' % d1), x_test_4)
    del x_test_4


if __name__ == '__main__':
    main()
