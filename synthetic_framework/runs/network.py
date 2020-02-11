import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
from numpy.linalg import inv
from runs.generate_data import DatasetGenerator
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

loss_dct = {'square_loss': 'mean_squared_error',
           'cross_entropy': 'categorical_crossentropy'}


class Network(tf.keras.Sequential):
    """ We generate here the class Network, which contains the different NN
    models implemented in the experiments. This class inherits from the tf
    Model class. Here we will just implement the different architectures.
    By default, as we generate the Network instance, we build the network. """

    def __init__(self, exp, data_path):
        """ Initializer of the Network class.
        We have the history which takes into account the performance
        during training and on the validation set.
        :param exp: an Experiment instance
        :param data_path: path where to recover the dataset
        """
        super(Network, self).__init__()

        self.exp = exp  # object of the Experiment class
        self.data_path = data_path  # string where the dataset is stored

        self.history = None  # history, needed only when we fit a model using tf
        self.trained = False  # turns True as we optimize
        self.to_fit = False  # True if not pseudo-inverse
        self._lr_reduction = None  # tf.callbacks
        self._early_stopping = None
        self.dct_kwargs = None    # we save the dictionary from the DatasetGenerator object
        self.test_loss = None
        self.test_accuracy = None

        data_gen = DatasetGenerator(data_path=self.data_path,
                                    load=True,
                                    key_dataset=self.exp.dataset.dataset_name,
                                    exp=self.exp)

        [X_splits, y_splits] = data_gen.generate_input_experiment()
        for (x_, y_) in zip(X_splits, y_splits):
            print(x_.shape, y_.shape)

        self.A = data_gen.A
        _, self.output_dims = y_splits[0].shape

        print('Input training dataset', X_splits[0].shape)

        self.build_network()  # here we build the network
        self.pseudo_inv = None
        self.reduce_dims = False
        self.optimize(X_splits[0], y_splits[0],
                      X_splits[1], y_splits[1])
        self.eval_metrics(X_splits[2], y_splits[2])
        self.save_outputs()


    def build_network(self):
        """
        We call this function during the initialization of the class.
        Here we generate the model. This inherits from tensorflow class.
        We define the architecture. The number of nodes for the hidden
        layer is fixed to the value nodes=128
        """
        if self.exp.hyper.architecture != 'pseudoinverse':
            self.to_fit = True

            # layer with ReLU architecture
            if self.exp.hyper.architecture == 'FC':
                nodes = 128
                self.add(Dense(nodes, activation='relu'))
            # linear layer
            elif self.exp.hyper.architecture == 'linear':
                nodes = 128
                self.add(Dense(nodes, activation='linear'))

            # cross entropy loss, we need the extra-layer
            if self.exp.hyper.loss == 'cross_entropy':
                self.add(Dense(self.output_dims,
                               activation='softmax'))
            if self.exp.hyper.loss == 'square_loss':
                self.add(Dense(self.output_dims,
                               activation='linear'))

            if self.exp.hyper.optimizer == 'sgd':
                sgd = SGD(lr=self.exp.hyper.learning_rate, momentum=0., nesterov=False)
            else:
                raise ValueError('This optimizer has not been included yet')

            if self.exp.hyper.lr_at_plateau:
                self._lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                       patience=5, min_lr=0)
            self._early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-6)

            self.compile(optimizer=sgd,
                         loss=loss_dct[self.exp.hyper.loss],
                         metrics=['accuracy'])

    def optimize(self, x_tr, y_tr, x_vl, y_vl):
        """ We train the algorithm.
        :param x_tr: training input data
        :param y_tr: training output values
        :param x_vl: validation input data
        :param y_vl: validation output data
        """
        if self.to_fit:
            callbacks_list = [self._lr_reduction,
                              self._early_stopping]
            history = self.fit(x_tr, y_tr,
                               epochs=self.exp.hyper.epochs,
                               batch_size=self.exp.hyper.batch_size,
                               validation_data=(x_vl, y_vl),
                               callbacks=callbacks_list)
            print(self.summary())
            self.history = history.history
        elif self.exp.hyper.architecture == 'pseudoinverse':
            # PSEUDO-INVERSE
            self.reduce_dims = False
            input_dimensions = x_tr.shape[1]  # number of features
            original_dims = input_dimensions - self.exp.dataset.additional_dims

            if self.exp.dataset.scenario == 1:
                if input_dimensions > self.exp.dataset.n_training:
                    # n < p, ill-posed problem
                    inv_val = inv(np.dot(x_tr, x_tr.T))  # this is n x n
                    self.pseudo_inv = np.dot(x_tr.T, np.dot(inv_val, y_tr))
                else:
                    inv_val = inv(np.dot(x_tr.T, x_tr))  # this is p x p
                    self.pseudo_inv = np.dot(inv_val, np.dot(x_tr.T, y_tr))

            elif self.exp.dataset.scenario == 2:
                # F (largest , smaller) dimensions
                F = np.vstack((np.identity(original_dims),
                               self.A[:self.exp.dataset.additional_dims, :]))  # frame in the redundant case
                if self.exp.dataset.n_training <= original_dims:  # n < p
                    frame_prod = np.dot(F.T, F)
                    # w = (X FT F XT)**(-1)
                    inv_val = inv(np.dot(np.dot(x_tr[:, :original_dims],
                                                frame_prod),
                                         x_tr[:, :original_dims].T))
                    # w = F XT (X FT F XT)**(-1) y
                    self.pseudo_inv = np.dot(F.dot(x_tr[:, :original_dims].T),
                                             np.dot(inv_val, y_tr))
                else:
                    # this is wF, n > p
                    # w F = (XsT Xs)**(-1) XsT y
                    self.pseudo_inv = np.dot(inv(np.dot(x_tr[:, :original_dims].T,
                                                        x_tr[:, :original_dims])),
                                             np.dot(x_tr[:, :original_dims].T, y_tr))
                    self.reduce_dims = True

            elif self.exp.dataset.scenario == 4:

                dims_noise_ = (int(np.ceil(self.exp.dataset.additional_dims *
                                           (1 - self.exp.dataset.redundancy_amount))))
                dims_redundant_ = (int(np.floor(self.exp.dataset.additional_dims *
                                                self.exp.dataset.redundancy_amount)))
                x_tr_ = x_tr[:, :input_dimensions-dims_redundant_]
                lw_part = np.hstack((self.A[:dims_redundant_, :],
                                     np.zeros((dims_redundant_, dims_noise_))))
                F = np.vstack((np.identity(dims_noise_ + original_dims),
                               lw_part))

                if self.exp.dataset.n_training <= input_dimensions-dims_redundant_:  # n < p
                    # w = F XNT (XN FT F XNT)**(-1) y
                    inv_val = inv(np.dot(x_tr_, np.dot(np.dot(F.T, F), x_tr_.T)))
                    self.pseudo_inv = np.dot(np.dot(F.dot(x_tr_.T), inv_val), y_tr)
                else:  # this is FTW  n > p
                    # w F = (XNT XN)**(-1) XNT y
                    self.pseudo_inv = np.dot(inv(np.dot(x_tr_.T, x_tr_)), np.dot(x_tr_.T, y_tr))
                    self.reduce_dims = True

        self.trained = True

    def eval_metrics(self, x_ts, y_ts):
        """ Performance evaluation after the optimize call.
        :param x_ts: input data, test set
        :param y_ts: output data, test set
        """
        if not self.trained:
            raise ValueError('The model has not been fitted yet')

        if self.to_fit:
            test_loss, test_accuracy = self.evaluate(x_ts, y_ts, verbose=2)

        else:  # pseudo-inverse prediction
            if self.reduce_dims:
                n_feat, _ = self.pseudo_inv.shape
                x_ts = x_ts[:, :n_feat]   # this is the number of independent features
            regression_pred = np.dot(x_ts, self.pseudo_inv)
            test_loss = (1/(x_ts.shape[0])) * np.sum((y_ts - regression_pred)**2)  # MSE

            y_pred = np.zeros_like(regression_pred).astype('int')
            y_pred[np.arange(y_pred.shape[0]), np.argmax(regression_pred, axis=-1)] = 1

            test_accuracy = np.sum(np.array([np.dot(y_t, y_p)
                                            for (y_t, y_p) in zip(y_ts, y_pred)])) / y_ts.shape[0]

        self.test_loss = test_loss
        self.test_accuracy = test_accuracy

    def save_outputs(self):
        """
        Save the content of the network.
        We save the weights and the history.
        We do not save the object Network, because it is redundant.
        """
        if self.to_fit:
            self.save_weights(join(self.exp.output_path,
                                   'weights.h5'),
                              save_format='h5')
            df = pd.DataFrame(data=self.history)
            df.to_csv(join(self.exp.output_path,
                           'history.csv'))
            np.save(join(self.exp.output_path, 'test.npy'), np.append(self.test_loss,
                                                                      self.test_accuracy))
            del df

        else:
            np.save(join(self.exp.output_path, 'weights.npy'),
                    self.pseudo_inv)
            np.save(join(self.exp.output_path, 'test.npy'),
                    np.append(self.test_loss,
                              self.test_accuracy))