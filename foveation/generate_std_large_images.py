import os
import numpy as np
import tensorflow as tf
from os.path import join
from PIL import Image


def add_noise_and_std(x, edge, noise_var=2e-1):
    """ Add noise to the original image and standardize the entire image.
    The pixels for this image are between values [0,1].
    :param x: original tensor
    :param edge: additional pixels
    :param noise_var: noise variance
    We generate the larger image, where the noise is such to be positive.
    We then standardize every image, so that its pixels distribution become Gaussian.
    """
    n_samples, dim1, dim2 = x.shape
    out = noise_var * np.abs(np.random.randn(n_samples, 2*edge+dim1, 2*edge+dim2))
    out[:, edge:edge+dim1, edge:edge+dim2] = x
    out_std = np.zeros_like(out)
    mean_ = np.mean(out, axis=(1, 2))
    std_ = np.std(out, axis=(1, 2))
    print(mean_.shape, std_.shape)
    print(out.shape)
    for k_, (m_, s_) in enumerate(zip(mean_, std_)):
        out_std[k_] = (out[k_] - m_) / s_
    return out_std


def upscale_std(x, new_shape_x=150, new_shape_y=150):
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
        tmp_x = image.resize(size=(new_shape_x, new_shape_y))
        tmp_std_x = (tmp_x - np.mean(tmp_x)) / np.std(tmp_x)
        new_x[n_] = tmp_std_x

    return new_x


def upscale_no_std(x, new_shape_x, new_shape_y):
    """ Upscale for experiment 4, no standardization yet.
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


def upscale_add_noise_std(x, new_dim, noise_var=2e-1, output_dim=500):
    """ Add noise to the original image.
    :param x: tensor of size n_samples, dim_x, dim_y
    :param new_dim:
    :param noise_var: noise variance, Gaussian noise
    :return:
    """
    # x is the original image -- here we upscale
    upscaled_mnist = upscale_no_std(x, new_shape_x=new_dim, new_shape_y=new_dim)
    edge = int((output_dim - new_dim) / 2)

    return add_noise_and_std(upscaled_mnist, edge)


def main():

    mnist = tf.keras.datasets.mnist  # load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    _, dim1, dim2 = x_test.shape

    edge_list = [146]

    folder_dataset = '/om/user/vanessad/foveation/modified_std_small_MNIST_dataset'
    os.makedirs(folder_dataset, exist_ok=True)

    # experiment 1
    """print('exp 1')
    folder_exp_1 = join(folder_dataset, 'exp_1_320')
    os.makedirs(folder_exp_1, exist_ok=True)
    for i_, e_ in enumerate([edge_list[-1]]):
        for kk, x_train_split in enumerate(np.split(x_train, 100, axis=0)):
            x_train_ = add_noise_and_std(x_train_split, e_)
            _, dim_x, dim_y = x_train_.shape
            np.save(join(folder_exp_1, 'exp_1_dim_%i_split_%i_tr.npy' % (dim_x, kk)), x_train_)
            del x_train_

    for i_, e_ in enumerate([edge_list[-1]]):
        for kk, x_test_split in enumerate(np.split(x_test, 100, axis=0)):
            x_test_ = add_noise_and_std(x_test_split, e_)
            _, dim_x, dim_y = x_test_.shape
            np.save(join(folder_exp_1, 'exp_1_dim_%i_split_%i_ts.npy' % (dim_x, kk)), x_test_)
            del x_test_"""

    print('exp 2')
    folder_exp_2 = join(folder_dataset, 'exp_2_320')
    os.makedirs(folder_exp_2, exist_ok=True)
    """for i_, e_ in enumerate([edge_list[-1]]):
        for kk, x_train_split in enumerate(np.split(x_train, 100, axis=0)):
            _, old_dim_x, old_dim_y = x_train_split.shape
            new_dim_x = old_dim_x + 2 * e_
            new_dim_y = old_dim_y + 2 * e_
            x_train_ = upscale_std(x_train_split, new_shape_x=new_dim_x, new_shape_y=new_dim_y)
            np.save(join(folder_exp_2, 'exp_2_dim_%i_split_%i_tr.npy' % (new_dim_x, kk)), x_train_)
            del x_train_"""

    for i_, e_ in enumerate([edge_list[-1]]):
        for kk, x_test_split in enumerate(np.split(x_test, 100, axis=0)):
            _, old_dim_x, old_dim_y = x_test.shape
            new_dim_x = old_dim_x + 2 * e_
            new_dim_y = old_dim_y + 2 * e_
            x_test_ = upscale_std(x_test_split, new_shape_x=new_dim_x, new_shape_y=new_dim_y)
            np.save(join(folder_exp_2, 'exp_2_dim_%i_split_%i_ts.npy' % (new_dim_x, kk)), x_test_)
            del x_test_

    print('exp 4')
    folder_exp_4 = join(folder_dataset, 'exp_4_320')
    os.makedirs(folder_exp_4, exist_ok=True)
    for i_, e_ in enumerate([edge_list[-1]]):
        for kk, x_train_split in enumerate(np.split(x_train, 100, axis=0)):
            _, old_dim, _ = x_train_split.shape
            new_dim = old_dim + 2 * e_
            x_train_4 = upscale_add_noise_std(x_train_split, new_dim=new_dim)
            np.save(join(folder_exp_4, 'exp_4_tr', 'exp_4_dim_%i_split_%i_tr.npy' % (e_, kk)), x_train_4)
            del x_train_4

    for i_, e_ in enumerate([edge_list[-1]]):
        for kk, x_test_split in enumerate(np.split(x_test, 100, axis=0)):
            _, old_dim, _ = x_test_split.shape
            new_dim = old_dim + 2 * e_
            x_test_4 = upscale_add_noise_std(x_test_split, new_dim=new_dim)
            np.save(join(folder_exp_4, 'exp_4_ts', 'exp_4_dim_%i_split_%i_ts.npy' % (e_, kk)), x_test_4)
            del x_test_4


if __name__ == '__main__':
    main()
