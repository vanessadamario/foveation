import numpy as np
from PIL import Image


class DatasetGenerator(object):
    def __init__(self, data, output_dim=28, scenario=1, noise_var=None, mnist_dim=None):
        """ DatasetGenerator initialization.
        :param data: original dataset, MNIST
        :param output_dim: the dimensionality for the output matrix
        :param scenario: one of the paradigm proposed [1, 2, 4]
        :param noise variance: useful in paradigm 1, 4
        :param original_mnist_dim: dimensionality before adding noise -- scenario 4
        """
        self.data = data
        self.output_dim = output_dim
        self.scenario = scenario
        self.output = None

        if noise_var is None:
            noise_var = 2e-1
        self.noise_var = noise_var

        n_samples, dim1, dim2 = self.data.shape
        # here we want to split

        self.n_samples = n_samples
        self.dim1 = dim1
        self.dim2 = dim2

        self.edge = int((self.output_dim - self.dim1) / 2)

        self.mnist_dim = mnist_dim  # we upscale and then add noise

        if self.scenario == 4 and self.mnist_dim is None:
            raise ValueError('The dimensionality of mnist with respect to the entire image must be defined')
        elif self.scenario == 4:
            self.edge = int((self.output_dim - self.mnist_dim) / 2)


    def add_noise_and_std(self):
        """ Add noise to the original image and standardize the entire image.
        The pixels for this image are between values [0,1].
        :param x: original tensor
        :param edge: additional pixels
        :param noise_var: noise variance
        We generate the larger image, where the noise is such to be positive.
        We then standardize every image, so that its pixels distribution become Gaussian.
        """
        out = self.noise_var * np.abs(np.random.randn(self.n_samples,
                                                      2 * self.edge + self.dim1,
                                                      2 * self.edge + self.dim2))
        out[:, self.edge:self.edge+self.dim1, self.edge:self.edge+self.dim2] = self.data

        out_std = np.zeros_like(out)
        mean_ = np.mean(out, axis=(1, 2))
        std_ = np.std(out, axis=(1, 2))
        for k_, (m_, s_) in enumerate(zip(mean_, std_)):
            out_std[k_] = (out[k_] - m_) / s_
        self.output = out_std
        return self


    def upscale_std(self):
        """
        Automatic PIL upscale of the image with standardization
        """
        new_x = np.zeros((self.n_samples, self.output_dim, self.output_dim))

        for n_, old_image_ in enumerate(self.data):
            image = Image.fromarray(old_image_)
            tmp_x = image.resize(size=(self.output_dim, self.output_dim))
            tmp_std_x = (tmp_x - np.mean(tmp_x)) / np.std(tmp_x)
            new_x[n_] = tmp_std_x

        self.output = new_x

        return self


    def _upscale_no_std(self):
        """ Upscale for experiment 4 wo standardization
        """
        new_x = np.zeros((self.n_samples, self.mnist_dim, self.mnist_dim))
        for n_, old_image_ in enumerate(self.data):
            image = Image.fromarray(old_image_)
            new_x[n_] = image.resize(size=(self.mnist_dim, self.mnist_dim))
        self.dim1 = self.mnist_dim
        self.dim2 = self.mnist_dim
        return new_x

    def upscale_add_noise_std(self):
        upscaled_mnist = self._upscale_no_std()
        self.data = upscaled_mnist
        self.add_noise_and_std()

    def run(self):
        if self.scenario == 1:
            self.add_noise_and_std()
        elif self.scenario == 2:
            self.upscale_std()
        elif self.scenario == 4:
            self.upscale_add_noise_std()
        else:
            raise ValueError('Nope')
