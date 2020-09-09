# this class is aimed at generating data for the redundant and noisy contexts
import os
import numpy as np
import pandas as pd
from os.path import join
import random
from runs.experiments import Experiment


def generate_tr_vl_ts_splits(id,
                             source_path,
                             split_path,
                             n_tr=3800,
                             n_vl=872,
                             n_ts=1821):
    """ Here we generate the train, validation,
    and test split starting from the training set of the SST2 only.
    The data are available at
                    https://github.com/CS287/HW1/tree/master/data
    We save the three splits into three different files
    :param id: the id of the experiment, not necessary for computation
    :param source_path: where to retrieve the original splits
    :param split_path: where to save the result
    :param n_vl: number of validation examples
    :param n_ts: number of test examples
    """
    train_sentences = pd.read_csv(join(source_path, 'train.csv'), index_col=0)
    train_labels = np.load(join(source_path, 'train.npy'))

    file_names = ['train', 'valid', 'test']
    div = np.array([n_tr // 2, n_vl // 2, n_ts // 2])
    splits = np.cumsum(div)
    seed = 10

    pos_id = np.argwhere(train_labels == 1).squeeze()
    neg_id = np.argwhere(train_labels == 0).squeeze()

    random.Random(seed).shuffle(pos_id)
    random.Random(seed).shuffle(neg_id)

    splits_pos = np.split(pos_id, splits)  # here we split in the three dataset
    splits_neg = np.split(neg_id, splits)
    for p_, n_, name_ in zip(splits_pos, splits_neg, file_names):
        df = pd.concat([train_sentences.loc[p_], train_sentences.loc[n_]])  # think if you want to change this
        size_, _ = df.shape
        df_ = pd.DataFrame(df.values, index=[k_ for k_ in range(size_)])
        y = np.append(train_labels[p_], train_labels[n_])
        df_.to_csv(join(split_path, name_ + '.csv'))
        np.save(join(split_path, name_ + '.npy'), y)
    return None


class DataGenerator:
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
                 n_max=None,
                 train=True,
                 split='train',
                 embedding_path=None):
        """ Initializer for the class. We pass the object Experiment to
        assess the transformation required. If we train, the maximum number of examples is required. If None
        we will transform the entire split.
        :param exp: Experiment object
        :param n_max: max amount of examples that we eventually want to transform
        :param train: bool, if True we generate the index for training, otherwise we load them
        :param split: str, the name of the split
        :param embedding_path: str, the name of the path for the three splits
        """
        self.exp = exp
        self.n_max = n_max
        self.train = train
        self.split = split
        self.embedding_path = embedding_path

        self.n_tr = None
        self.y_split = None

        if train:
            self.generate_minimal()

    def generate_minimal(self):
        """ This n_training corresponds to the maximum amount of training examples.
        """
        fold_ = join(os.path.dirname(self.exp.output_path), 'train_indexes')
        os.makedirs(fold_, exist_ok=True)
        ind_filename = 'train_indexes.npy'
        path_tr_ = join(self.embedding_path, 'train_sentences')

        y_train_len = len(os.listdir(path_tr_)) // 2
        if ind_filename not in os.listdir(fold_):
            id_tr_pos = []
            id_tr_neg = []

            while len(id_tr_pos) < self.n_max // 2 or len(id_tr_neg) < self.n_max // 2 :
                id_ = np.random.choice(np.arange(y_train_len))
                if id_ not in id_tr_pos or id_ not in id_tr_neg:
                    if 'y_%i.npy' % id_ in os.listdir(path_tr_):
                        y_ = np.load(join(path_tr_, 'y_%i.npy' % id_))
                        if y_ == 1 and len(id_tr_pos) < self.n_max // 2:
                            id_tr_pos.append(id_)
                        elif y_ == 0 and len(id_tr_neg) < self.n_max // 2:
                            id_tr_neg.append(id_)

            n_tr = np.array([], dtype=int)
            y_split = np.array([], dtype=int)
            for p_, n_ in zip(id_tr_pos, id_tr_neg):
                n_tr = np.append(n_tr, p_)
                n_tr = np.append(n_tr, n_)
                y_split = np.append(y_split, np.array([1, 0]))

            np.save(join(fold_, 'train_indexes.npy'), n_tr)  # length n_max
            # the amount of redundancy can change across experiments
            self.y_split = y_split
            self.n_tr = n_tr
        else:
            self.n_tr = np.load(join(fold_, ind_filename))
            self.y_split = np.array([np.load(join(path_tr_, 'y_%i.npy' % n_)) for n_ in self.n_tr])

    def generate_x_y(self):
        """ Generate the input - output dataset, as we pass the indexes.
        Here we consider as input one of the three splits. If the split from the
        validation or the test set, then we will consider all the samples in it.
        If the split is from the training, we will take into account only the
        number of training samples specific for the experiment.
        In this last case, the training maps are divided, and we consider

        """
        iter_ = self.n_tr[:self.exp.dataset.n_training] if self.split == 'train' \
            else np.arange(len(os.listdir(join(self.embedding_path,
                                               self.split + '_sentences')))//2)

        x_lst = []
        y_array = np.array([])
        for i_ in iter_:
            if 'y_%i.npy' % i_ in os.listdir(join(self.embedding_path, self.split + '_sentences')):
                y_array = np.append(y_array,
                                    np.load(join(self.embedding_path, self.split + '_sentences', 'y_%i.npy' % i_)))
                x_ = np.load(join(self.embedding_path, self.split + '_sentences', 'sentence_%i.npy' % i_))

                if self.exp.dataset.redundant_phrases > 0:
                    path_r = join(self.embedding_path, self.split + '_phrases', 'phrases_r')
                    files_ = [f_ for f_ in os.listdir(path_r) if f_.startswith('map_%i_' % i_)]
                    n_files = len(files_)
                    for f_ in range(int(n_files * self.exp.dataset.redundant_phrases)):
                        x_ = np.vstack((x_, np.load(join(path_r, files_[f_]))))

                if self.exp.dataset.noisy_phrases > 0:
                    path_n = join(self.embedding_path, self.split + '_phrases', 'phrases_n')
                    files_ = [f_ for f_ in os.listdir(path_n) if f_.startswith('map_%i_' % i_)]
                    n_files = len(files_)
                    for f_ in range(int(n_files * self.exp.dataset.noisy_phrases)):
                        x_ = np.vstack((x_, np.load(join(path_n, files_[f_]))))
                x_lst.append(x_)

        y_one_hot = np.zeros((y_array.size, 2))
        y_one_hot[np.arange(y_array.size), y_array.astype(int)] = 1

        if self.exp.hyper.architecture == 'CNN':
            return x_lst, y_one_hot
        else:
            return np.array([np.mean(x_stc, axis=0) for x_stc in x_lst]), y_one_hot