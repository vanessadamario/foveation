import numpy as np
import pandas as pd
import tensorflow as tf
from os.path import join
import gensim.downloader as api
from runs.generate_data import DatasetGenerator

word_vectors = api.load("glove-wiki-gigaword-100")


class ShallowCNNTextClassifier:
    """ Here we generate a shallow CNN for text classification as in the work of Yoon Kim.

                https://arxiv.org/pdf/1408.5882.pdf

    The authors use convolutional filters of different sizes, respectively 3, 4, 5.
    The implementation of this network relies on TensorFlow2.
   """
    def __init__(self,
                 exp,
                 output_classes=2):
        """ We pass the experiment specifics and the number of classes,
        which corresponds to 2 for the sentiment analysis example here. """

        self.exp = exp
        self.window_size = [3, 4, 5]
        self.embedding = self.exp.dataset.embedding_dim
        self.kernels = self.exp.hyper.nodes   # kernels for each convolutional filter?
        self.output_classes = output_classes  # task of sentiment analysis
        self.lr = self.exp.hyper.learning_rate
        self.output_path = self.exp.output_path

        shapes_conv_layer = [[w_, self.embedding, 1, self.kernels] for w_ in self.window_size]
        shapes_last_layer = [[self.kernels * len(self.window_size), self.output_classes]]
        shapes = shapes_conv_layer + shapes_last_layer

        weights = [self.get_weight(shapes[i], 'weight{}'.format(i))
                   for i in range(len(shapes))]
        self.weights = weights

        bias_shapes = [[1, self.kernels] for _ in range(len(self.window_size))] + [[1, self.output_classes]]
        bias = [tf.Variable(np.zeros((b_[0], b_[1]), dtype=np.float32))
                for b_ in bias_shapes]
        self.bias = bias

        self.optimizer = tf.optimizers.SGD(learning_rate=self.lr, momentum=0.)
        self.best_early = np.Inf
        self.best_plateau = np.Inf
        self.current = None
        self.wait_lr = 0
        self.wait_early = 0
        self.stop_training = False

        # self.train_and_save()

    def get_weight(self, shape, name):
        """ Weights initializer.

        :param shape: the shape of the model, list of dimensions for the layer
        :param name: str, layer's name

        :returns tf.Variable: a tensor object corresponding to a trainable layer
        """
        initializer = tf.initializers.glorot_uniform()
        return tf.Variable(initializer(shape),
                           name=name,
                           trainable=True,
                           dtype=tf.float32)

    def _gen_padding_bm(self,
                        data):
        """ Here we create the input of the convolutional net.
        There are two matrices which are passed as input,
            the data, padded in such a way to have the shape of a tensor
            the boolean mask, which is needed to reduce edge effect

        :param data: this is a list, containing elements which are np.arrays
        of dimension (number_of_words, size_of_the_embedding)

        :returns data_padded: np.array of dimension (n_samples, max_len, emb, 1),
        with 1 corresponding to the number of channels
        :returns bool_mask: np.array of dimension (n_samples, max_len-window+1, emb, 1)
        1 correspond to the features we want to keep
        """
        n_samples = len(data)   # number of examples
        _, emb = data[0].shape  # dimensionality of the embedding
        max_length = np.max(np.array([d_.shape[0] for d_ in data]))
        data_padded = np.zeros((n_samples, max_length, emb, 1))  # uniform dim
        id_before_padding = [x_.shape[0] for x_ in data]

        for i_, (n_w_, x_) in enumerate(zip(id_before_padding, data)):
            data_padded[i_, :n_w_, :, :] = x_.reshape(n_w_, -1, 1)

        return data_padded

    def model(self, x, padded=False, eval=True):  # , list_lst_indices=None, lst_size_bm=None):
        """ We first reduce the dataset to a tensor of fixed dimensions.
        We moreover generate a boolean array so to discard edge effects.
        The architecture is such that

            padding, input tensor (n_samples, max_length, embedding, 1)
            bool_mask, input (n_samples, max_length-window+1, 1, kernels)

            o1 = w1 * padding, dim: (n, max_length-window+1, 1, kernels)
            o1 = o1 x bool_mask, dim: (n, max_length-window+1, 1, kernels)
            o1 = max(o1), dim: (n, kernels)
            o1 = w2 x o1, dim: (n, classes)
            o1 = softmax(o1), dim: (n, classes)

        :param x: input data
        :param padded: bool, if True the data have already been padded
        :param eval: bool, if False, we apply bernoulli dropout, otherwise we rescale the weights for the probability

        :returns logits: the output of the model, after applying the softmax function
        """
        padded_data = x if padded else self._gen_padding_bm(x)
        max_norm = 3
        bernoulli = 0.5
        x1 = tf.cast(padded_data, dtype=tf.float32)
        c_lst = [tf.nn.conv2d(x1,
                              self.weights[j_],
                              strides=1,
                              padding=[[0, 0],
                                       [0, 0],
                                       [0, 0],
                                       [0, 0]])
                 for j_ in range(len(self.weights)-1)]

        depad_lst = [tf.nn.relu(c_ + b_)
                     for (c_, b_) in zip(c_lst, self.bias[:-1])]

        max_pool_output = [tf.math.reduce_max(depad_, axis=(1, 2)) for depad_ in depad_lst]
        stack_output = tf.concat(max_pool_output, axis=-1)  # max pooling for all filters
        norm_factor = max_norm / tf.maximum(max_norm, tf.norm(self.weights[-1], axis=0))
        weights_output_ = tf.math.multiply(self.weights[-1], tf.expand_dims(norm_factor, axis=0))

        if eval:
            return tf.nn.softmax(tf.nn.relu(tf.matmul(stack_output, 0.5 * weights_output_)
                                            + self.bias[-1]))
        else:
            return tf.nn.softmax(tf.nn.relu(tf.matmul(tf.nn.dropout(stack_output, rate=bernoulli),
                                                        weights_output_)
                                            + self.bias[-1]))

    def train_epoch_step(self,
                         inputs,
                         outputs,
                         batch_size=50):
        """ Generate the batches and call the train_step_batch function.

        :param inputs: not uniform input, a list
        :param outputs: target, the labels
        :param batch_size: number of example for the batch

        :return [loss, accuracy]: loss and accuracy on the training set
        """
        n_samples = len(inputs)  # this is a list
        batches_per_epoch = int(np.floor(n_samples / batch_size))
        id_batches = np.random.choice(np.arange(n_samples),
                                      size=(batches_per_epoch, batch_size))
        for id_ in id_batches:
            tmp_in = self._gen_padding_bm([inputs[i__] for i__ in id_])
            self.train_batch_step(tmp_in, outputs[id_])
        return self.evaluate(inputs, outputs)

    def train_batch_step(self, padded_data, outputs):
        """
        Here we train on a single batch

        :param padded_data: np.array of uniform dimensions containing the embedding
        # :param bm_data: np.array with the bool mask to reduce the padding
        :param outputs: output labels

        :return: loss value over single batch
        """
        with tf.GradientTape() as tape:
            current_loss_sum_ = tf.reduce_mean(tf.losses.categorical_crossentropy(self.model(padded_data,
                                                                                             padded=True,
                                                                                             eval=False),
                                                                                  tf.one_hot(outputs,
                                                                                             depth=2)))
        grads = tape.gradient(current_loss_sum_, self.weights + self.bias)
        self.optimizer.apply_gradients(zip(grads, self.weights + self.bias))

    def evaluate(self, inputs, outputs):
        """
        This function evaluates the performance on the model of any type of set

        :param inputs: dataset, it can be both already padded or before padding
        :param outputs: the labels related to the inputs_data

        :return: the mean for the loss and the mean for the accuracy
        """
        n_samples = len(inputs)
        evaluation_batch = 50  # maximum tolerance on the sample size
        n_batches_eval = np.arange(0, n_samples, evaluation_batch) if n_samples > evaluation_batch else np.array([n_samples])
        loss_lst = []
        accuracy_lst = []

        for id_batch_, start_batch_ in zip(np.arange(n_batches_eval.size), n_batches_eval):
            end_batch_ = None if id_batch_ == n_batches_eval.size - 1 else start_batch_ + evaluation_batch
            logits = self.model(inputs[start_batch_:end_batch_])
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(logits,
                                                                     tf.one_hot(outputs[start_batch_:end_batch_],
                                                                                depth=2)))
            accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(tf.one_hot(outputs[start_batch_:end_batch_],
                                                                                  depth=2),
                                                                       tf.one_hot(tf.argmax(logits, 1),
                                                                                  depth=2)))
            loss_lst.append(loss)
            accuracy_lst.append(accuracy)
        return tf.reduce_mean(loss_lst), tf.reduce_mean(accuracy_lst)

    def fit(self,
            x_train,
            y_train,
            x_valid,
            y_valid,
            max_epochs=5,
            batch_size=32):
        """ Fit function.

        :param x_train: training input set, a list
        :param y_train: np.array, training labels
        :param x_valid: validation input set, a list
        :param y_valid: np.array, validation labels
        :param max_epochs: maximum number of epochs
        :param batch_size: dimensionality of the batch size
        """
        for e_ in range(max_epochs):
            print('Epoch %i' % e_)
            tmp_t_loss, tmp_t_acc = self.train_epoch_step(x_train, y_train, batch_size)
            tmp_v_loss, tmp_v_acc = self.evaluate(x_valid, y_valid)

            self.current = tmp_v_loss
            tmp_array = np.array([tmp_t_loss.numpy(), tmp_t_acc.numpy(), tmp_v_loss.numpy(), tmp_v_acc.numpy(), self.lr])
            if e_ == 0:
                output_vals = tmp_array
            else:
                output_vals = np.vstack((output_vals, tmp_array))
            self.lr_reduce_on_plateau()
            self.early_stopping()
            if self.stop_training:
                self.generate_history(output_vals)
                return
        self.generate_history(output_vals)

    def lr_reduce_on_plateau(self,
                             red_factor=0.1,
                             patience=5,
                             min_delta=1e-4,
                             min_lr=0.):
        """
        Monitor the validation loss so to have a schedule on the learning rate.
        We start by considering the loss at validation equivalent to infinity.
        We reduce its value is for patience epochs there is no improvement.

        :param red_factor: reduction factor
        :param patience: max the number of epochs before applying a change to the learning rate
        :param min_delta: the tolerance over the observed values
        :param min_lr: minimum learning rate value
        """
        if np.less(self.current, self.best_plateau - min_delta):  # if the
            self.best_plateau = self.current
            self.wait_lr = 0
        else:
            self.wait_lr += 1

            if self.wait_lr >= patience:
                old_lr = float(self.lr)
                if old_lr > min_lr:
                    new_lr = old_lr * red_factor
                    self.lr = max(new_lr, min_lr)
                    self.wait_lr = 0

    def early_stopping(self,
                       patience=10,
                       min_delta=1e-6):
        """
        Monitor the validation loss and stop training if there is no decrease of the validation loss after patience.

        :param patience: number of iterations after which, if there is no improvement, we stop
        :param min_delta: max value of the variation from which we stop the training
        """
        if np.less(self.current, self.best_early - min_delta):  # if it is decreasing
            self.best_early = self.current  # highest loss becomes the current one
            self.wait_early = 0  # wait is a variable related to the number of iterations
        else:
            self.wait_early += 1
            if self.wait_early >= patience:
                self.stop_training = True  # we stop training if this condition is verified
        return self.stop_training

    def generate_history(self, output_vals):
        """
        We generate a history.csv file containing the values for the training loss, accuracy,
        validation_loss, validation_accuracy and learning rates.

        :param output_vals: the matrix containing all the metrics computed during training and the learning rate
        value across iterations

        :return: pd.DataFrame with different entries
                 [epochs, loss, accuracy, validation_loss, validation_accuracy, lr]
        """
        df = pd.DataFrame(data=output_vals,
                          columns=['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr'])
        df.to_csv(join(self.exp.output_path, 'history.csv'))

    def train_and_save(self):
        """ Method to run and save network and results.
        We generate the data, as specified in self.exp,
        we give as validation samples 1000 points,
        we train, with a maximum number of 500 epochs, then we save
        the history and the test performance."""
        datagen = DatasetGenerator(self.exp, n_vl=1000)
        x_tr = datagen.lst_tr
        x_vl = datagen.lst_vl
        x_ts = datagen.lst_ts

        y_tr = datagen.y_tr
        y_vl = datagen.y_vl
        y_ts = datagen.y_ts

        self.fit(x_tr,
                 y_tr,
                 x_vl,
                 y_vl,
                 batch_size=self.exp.hyper.batch_size)

        ls_ts, ac_ts = self.evaluate(inputs=x_ts, outputs=y_ts)

        np.save(join(self.exp.output_path, 'test.npy'),
                np.array([ls_ts.numpy(), ac_ts.numpy()]))