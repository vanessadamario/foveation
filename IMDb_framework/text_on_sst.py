import os
import gensim
import tensorflow as tf
from os.path import join
from runs.experiment import Experiment
from tensorflow.optimizer import Adadelta
from runs.CNN_text_classification import ShallowCNNTextClassifier
import pandas as pd


def load_data():
    """ Here we need to load the data, we will use SST 2, and the representation obtained using word2vec."""


def main():

    path_to_file = '/om/user/vanessad/IMDb_framework/stanfordSentimentTreebank'

    learning_test_split = pd.read_csv(join(path_to_file, 'datasetSplit.txt'), index_col=0, header=0)

    # contains all the phrases and their IDs
    dictionary = pd.read_csv(join(path_to_file, 'dictionary.txt'), sep='|', header=None)

    # contains the original text
    rt_dataset = pd.read_csv(join(path_to_file, 'original_rt_snippets.txt'), sep='\n', header=None)

    # labels ID | sentiment
    # [0, 0.4] -- negative class, (0.6, 1.0] positive class
    labels = pd.read_csv(join(path_to_file, 'sentiment_labels.txt'), sep='|', header=0, index_col=0)
    print(labels)

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('/om/user/vanessad/IMDb_framework/GoogleNews-vectors-negative300.bin', binary=True)
    print(model.vocabulary)

    splits = dictionary[0][6000].split()
    print(splits)

    for s_ in splits:
        print(s_)

    exp = Experiment()
    exp.hyper.learning_rate = 5e-2
    exp.hyper.nodes = 100
    exp.hyper.window = [3, 4, 5]
    exp.hyper.epochs = 500
    exp.hyper.batch_size = 50
    exp.hyper.loss = 'cross_entropy'
    exp.hyper.lr_at_plateau = False

    exp.dataset.n_training = 10
    exp.dataset.embedding_dim = 300

    cnn_text = ShallowCNNTextClassifier(exp)
    cnn_text.optimizer = Adadelta()

