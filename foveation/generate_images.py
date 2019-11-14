import os
import numpy as np
import tensorflow as tf
from os.path import join
from PIL import Image


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
    """ We augment the original size of each image, by putting it in the center
    :param x: original image, tensor of size n_samples, dim_x, dim_y
    :param edge: number of pixels we want to enlarge the image for each size
    :return out: enlarged tensor
    """
    n_samples, dim1, dim2 = x.shape
    out = np.zeros((n_samples, 2*edge+dim1, 2*edge+dim2))
    out[:, edge:edge+dim1, edge:edge+dim2] = x
    return out


def augment_size_add_noise(x, edge, noise_var=2e-1):
    """ Add noise to the original image.
    :param x: tensor of size n_samples, dim_x, dim_y
    :param edge: number of pixels we want to enlarge the image for each size of the image
    :param noise_var: noise variance, Gaussian noise
    :return:
    """
    n_samples, dim1, dim2 = x.shape
    out = noise_var * np.abs(np.random.randn(n_samples, 2*edge+dim1, 2*edge+dim2))
    out[:, edge:edge+dim1, edge:edge+dim2] = x
    return out


def upscale(x, new_shape_x=150, new_shape_y=150):
    """
    Automatic PIL upscale of the image
    :param x: input data, typically the dataset, of three dimensions
    :param new_shape_x: int new_dim_x
    :param new_shape_y: int new_dim_y
    :return: new_x: new tensor
    """
    n_samples, dim1, dim2 = x.shape  # original
    new_x = np.zeros((n_samples, new_shape_x, new_shape_y))

    for n_, old_image_ in enumerate(x):
        image = Image.fromarray(old_image_)
        new_x[n_] = image.resize(size=(new_shape_x, new_shape_y))

    return new_x


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
