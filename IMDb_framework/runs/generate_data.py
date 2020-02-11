import numpy as np
import gensim.downloader as api
from tensorflow.keras.datasets import imdb

word_vectors = api.load("glove-wiki-gigaword-100")


class DatasetGenerator:
    """ This class is meant to be the generator for different
    types of transformation to the IMDb data. We load the IMDb
    dataset and we apply different transformation, depending on the
    Experiment attributes.
    The training samples are extracted from the exp class
    The validation samples per class are 5000 by default.
    The test samples are the one in the original test set.
    The extraction is such that we get a balanced dataset.
    """

    def __init__(self,
                 exp,
                 n_vl=5000):
        """ Initializer for the class. We pass the object Experiment to
        assess the transformation required.
        :param exp: Experiment object
        :param n_vl: int, number of validation samples per class
        """
        self.exp = exp
        self.n_vl = n_vl

        (tr_data, tr_labels), (ts_data, ts_labels) = imdb.load_data(num_words=5000,
                                                                    index_from=0)
        word_index = imdb.get_word_index(path='imdb_word_index.json')
        self.reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        self.glove_embedding = word_vectors

        # remove the indexes based on start_
        id_tr, id_vl = self._split_train_validation(tr_labels)

        mean_tr, lst_tr = self.output_embedding([tr_data[i] for i in id_tr])
        self.mean_tr = mean_tr
        self.lst_tr = lst_tr
        self.y_tr = tr_labels[id_tr]

        mean_vl, lst_vl = self.output_embedding([tr_data[i] for i in id_vl])
        self.mean_vl = mean_vl
        self.lst_vl = lst_vl
        self.y_vl = tr_labels[id_vl]

        mean_ts, lst_ts = self.output_embedding([x_ts for x_ts in ts_data])
        self.mean_ts = mean_ts
        self.lst_ts = lst_ts
        self.y_ts = ts_labels

    def _split_train_validation(self, y_learning):
        """ Split of the training and validation set.
        We chose randomly n_training elements from the learning set.
        :param y_learning: labels from the training IMDb dataset.
        """
        id_tr, id_vl = np.array([], dtype=int), np.array([], dtype=int)
        n_tr = self.exp.dataset.n_training

        for y_ in np.unique(y_learning):
            id_class_y_ = np.where(y_learning == y_)[0]
            tmp_id_tr = np.random.choice(id_class_y_,
                                         size=n_tr,
                                         replace=False)
            tmp_id_vl = np.random.choice(np.setdiff1d(id_class_y_, tmp_id_tr),
                                         size=self.n_vl,
                                         replace=False)
            id_tr = np.append(id_tr, tmp_id_tr)
            id_vl = np.append(id_vl, tmp_id_vl)
        return id_tr, id_vl

    def output_embedding(self, X):
        """ Dataset, it has dimensions (n, #words in sample i-th)
        :param X: dataset, list of length n (samples), containing lists
        :return mean_set: the mean embedding for each sample
        :return lst_set: list containing the embedding for each word
        """
        mean_set, lst_set = [], []
        for x in X:
            mean_, lst_ = self.preprocessing(x)  # x is a sample (containing n indexes)
            mean_set.append(mean_)
            lst_set.append(lst_)
        return np.array(mean_set), lst_set

    def preprocessing(self, x):
        """ Here we call the functions to perform different types of pre-processing.
        This consists in:
            1) excluding the most frequent words of the dictionary
            2) remove a fixed amount of words
            3) transform indexes into words
            4) embed the each word and average
        :param x: a sample, which is a list containing different indexes,
        one for each word
        """
        x = self._exclude_most_freq(x)
        x = self._exclude_words(x)
        x = self._index2str(x)
        return self._embedding(x)

    def _exclude_most_freq(self, x):
        """ We exclude the most frequent words here.
        :param x: we pass one sample, a list containing indexes
        """
        x = np.array(x)
        return x[x >= self.exp.dataset.first_index]

    def _exclude_words(self, x):
        """ We exclude words here.
        The most frequent ones are typically not relevant to the task.
        :param x: a sample, it contains the indexes for the sample
        """
        if self.exp.dataset.removed_words == 0:
            return x
        elif self.exp.dataset.removed_words >= 1:
            raise ValueError("Maximum value of removed words must be less than one.")
        n_to_rm = int(self.exp.dataset.removed_words * len(x))
        rnd_rm = np.random.choice(np.arange(len(x)), size=n_to_rm)
        return list(np.delete(np.array(x), rnd_rm))

    def _index2str(self, x):
        """ From indices to words, given a single sample.
        Transform in a list of string values.
         :param x: a data from imdb.load_data(),
         it contains the most used words
         """
        return [self.reverse_word_index[id_] for id_ in x]

    def _embedding(self, words_lst):
        """ Here we generate the embedding for each sample.
        We use the pre-trained word-vectors from gensim-data
        :param words_lst: the list of words in a sample
        :return embedding_mean: the mean value for the embedding
        over the entire sample.
        :return embedding_array: for each sample, this is a matrix,
        each row is the embedding of a word in the dataset
        """
        emb_lst = []
        for w_ in words_lst:
            if w_ in self.glove_embedding.vocab:
                emb_lst.append(self.glove_embedding[w_])
        embedding_array = np.array(emb_lst)
        embedding_mean = np.mean(embedding_array, axis=0)
        return embedding_mean, embedding_array