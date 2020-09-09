import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
from tensorflow.keras.layers import Flatten, Dense, Conv2D, GlobalMaxPooling2D
from tensorflow.keras.optimizers import SGD, Adadelta
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

loss_dct = {'cross_entropy': 'categorical_crossentropy'}
activation_dct = {'FC': 'relu',
                  'CNN': 'relu'}
activation_lss = {'cross_entropy': 'softmax'}


class Network(tf.keras.Sequential):
    """ We generate here the class Network, which contains the different NN
    models implemented in the experiments. This class inherits from the tf
    Model class. Here we will just implement the different architectures.
    By default, as we generate the Network instance, we build the network. """

    def __init__(self, exp):
        """ Initializer of the Network class.
        We have the history which takes into account the performance
        during training and on the validation set.
        :param exp: an Experiment instance
        """
        super(Network, self).__init__()

        self.exp = exp  # object of the Experiment class

        self.history = None  # history, needed only when we fit a model using tf
        self.trained = False  # turns True as we optimize
        self._lr_reduction = None  # tf.callbacks
        self._early_stopping = None
        self.test_loss = None
        self.test_accuracy = None

        self.build_network()  # here we build the network

        """
        data_gen = DatasetGenerator(self.exp)
        y_tr, y_vl, y_ts = data_gen.y_tr, data_gen.y_vl, data_gen.y_ts

        x_tr = data_gen.mean_tr
        x_vl = data_gen.mean_vl
        x_ts = data_gen.mean_ts
        _, embedding_size = x_tr.shape
        self.embedding_size = embedding_size

        del data_gen
        self.optimize(x_tr, y_tr,
                      x_vl, y_vl)

        del x_tr, x_vl, y_tr, y_vl

        self.eval_metrics(x_ts, y_ts)
        self.save_outputs()"""

    def build_network(self):
        """
        We call this function during the initialization of the class.
        Here we generate the model. This inherits from tensorflow class.
        We define the architecture. The number of nodes for the hidden
        layer is fixed to the value nodes=128
        """
        # for linear and fully connected with ReLU architecture
        self.add(Dense(self.exp.hyper.nodes, activation=activation_dct[self.exp.hyper.architecture]))

        self.add(Dense(self.exp.dataset.output_dims, activation=activation_lss[self.exp.hyper.loss]))

        if self.exp.hyper.optimizer == 'SGD':
            opt = SGD(lr=self.exp.hyper.learning_rate, momentum=0., nesterov=False)
        elif self.exp.hyper.optimizer == 'adadelta':
            opt = Adadelta(lr=self.exp.hyper.learning_rate)
        else:
            raise ValueError('This optimizer has not been included yet')

        if self.exp.hyper.lr_at_plateau:
            self._lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                   patience=5, min_lr=0)
        self._early_stopping = EarlyStopping(monitor='val_loss', patience=8, min_delta=1e-4)
        # min_delta 1e-6 for sst2, patience 10

        self.compile(optimizer=opt,
                     loss=loss_dct[self.exp.hyper.loss],
                     metrics=['accuracy'])

    def optimize(self, x_tr, y_tr, x_vl, y_vl):
        """ We train the algorithm.
        :param x_tr: training input data
        :param y_tr: training output values
        :param x_vl: validation input data
        :param y_vl: validation output data
        """
        callbacks_lst = [self._lr_reduction,
                         self._early_stopping] if self.exp.hyper.lr_at_plateau else [self._early_stopping]
        print(x_tr)
        print('type', type(x_tr))

        history = self.fit(x_tr, y_tr,
                           epochs=self.exp.hyper.epochs,
                           batch_size=self.exp.hyper.batch_size,
                           validation_data=(x_vl, y_vl),
                           callbacks=callbacks_lst)

        print(self.summary())
        self.history = history.history
        self.trained = True

    def test_and_save(self, x_ts, y_ts):
        """ Performance evaluation after the optimize call.
        :param x_ts: input data, test set
        :param y_ts: output data, test set
        """
        if not self.trained:
            raise ValueError('The model has not been fitted yet')

        if self.trained:
            test_loss, test_accuracy = self.evaluate(x_ts, y_ts, verbose=2)

            self.test_loss = test_loss
            self.test_accuracy = test_accuracy
        self.save_weights(join(self.exp.output_path,
                               'weights.h5'),
                          save_format='h5')
        df = pd.DataFrame(data=self.history)
        df.to_csv(join(self.exp.output_path,
                       'history.csv'))
        np.save(join(self.exp.output_path, 'test.npy'), np.append(self.test_loss,
                                                                  self.test_accuracy))
        del df

    def save_outputs(self):
        """ Save the content of the network.
        We save the weights and the history.
        We do not save the object Network, because it is redundant.
        """
        self.save_weights(join(self.exp.output_path,
                               'weights.h5'),
                          save_format='h5')
        df = pd.DataFrame(data=self.history)
        df.to_csv(join(self.exp.output_path,
                       'history.csv'))
        np.save(join(self.exp.output_path, 'test.npy'), np.append(self.test_loss,
                                                                  self.test_accuracy))
        del df