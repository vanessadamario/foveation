# this class is aimed at generating data for the redundant and noisy contexts
import os
import gensim
from runs.experiments import Experiment
import numpy as np
import pandas as pd
from os.path import join
import random


model = gensim.models.KeyedVectors.load_word2vec_format(
    '/om/user/vanessad/IMDb_framework/GoogleNews-vectors-negative300.bin', binary=True)
# split_path = '/om/user/vanessad/IMDb_framework/split_train_SST2'
# phrases_path = '/om/user/vanessad/IMDb_framework/HW1'
phrases_per_map = 10


def generate_tr_vl_ts_splits(id,
                             source_path,
                             split_path,
                             n_tr=3800,
                             n_vl=872,
                             n_ts=1821):
    """ Here we generate the train, validation, and test split starting from the training set of the SST2 only.
    The data are available at
                    https://github.com/CS287/HW1/tree/master/data
    We save the three splits into three different files
    :param id: the id of the experiment, not necessary for computation
    :param source_path: where to retrieve the original splits
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
                 split_path=split_path,
                 phrases_path=phrases_path):
        """ Initializer for the class. We pass the object Experiment to
        assess the transformation required. If we train, the maximum number of examples is required. If None
        we will transform the entire split.
        :param exp: Experiment object
        :param n_max: max amount of examples that we eventually want to transform
        :param train: bool, if True we generate the index for training, otherwise we load them
        :param split: str, the name of the split
        :param split_path: str, the name of the path for the three splits
        :param phrases_path: str, the name of the path to the phrases
        """
        self.min_length = 5
        self.exp = exp
        self.n_max = n_max
        self.train = train
        self.split = split
        self.split_path = split_path
        self.phrases_path = phrases_path

        self.noise_phrases = []
        self.red_phrases = []
        self.n_tr = None

        if train:
            self.generate_minimal()
        else:  # in case of validation or test splits
            self.x_split = pd.read_csv(join(self.split_path, self.split + '.csv'), index_col=0)
            self.y_split = np.load(join(self.split_path, self.split + '.npy'))

    def generate_minimal(self):
        """ This n_training corresponds to the maximum amount of training examples.
        """
        fold_ = join(os.path.dirname(self.exp.output_path), 'train_indexes')
        os.makedirs(fold_, exist_ok=True)
        ind_filename = 'train_indexes.npy'

        train_x = pd.read_csv(join(self.split_path, 'train.csv'), index_col=0)
        train_y = np.load(join(self.split_path, 'train.npy'))

        if ind_filename not in os.listdir(fold_):

            id_tr_pos = []
            id_tr_neg = []

            while len(id_tr_pos) < self.n_max // 2:
                tmp_id = np.random.choice(np.argwhere(train_y == 1).squeeze())
                if tmp_id in id_tr_pos:
                    continue
                tmp_x = [f_ for f_ in train_x.loc[tmp_id] if isinstance(f_, str)]
                if len(tmp_x) >= self.min_length:
                    id_tr_pos.append(tmp_id)
            while len(id_tr_neg) < self.n_max // 2:
                tmp_id = np.random.choice(np.argwhere(train_y == 0).squeeze())
                if tmp_id in id_tr_neg:
                    continue
                tmp_x = [f_ for f_ in train_x.loc[tmp_id] if isinstance(f_, str)]
                if len(tmp_x) >= self.min_length:
                    id_tr_neg.append(tmp_id)

            n_tr = np.array([], dtype=int)
            for p_, n_ in zip(id_tr_pos, id_tr_neg):
                n_tr = np.append(n_tr, p_)
                n_tr = np.append(n_tr, n_)

            np.save(join(fold_, 'train_indexes.npy'), n_tr)  # length n_max
            # the amount of redundancy can change across experiments
            self.x_split = train_x.loc[n_tr]
            self.y_split = train_y[n_tr]
            self.n_tr = n_tr
        else:
            self.n_tr = np.load(join(fold_, ind_filename))
            self.x_split = train_x.loc[self.n_tr]
            self.y_split = train_y[self.n_tr]

    def flatten(self, lst):
        """Flatten a list."""
        return [y for l in lst for y in self.flatten(l)] \
            if isinstance(lst, (list, np.ndarray)) else [lst]

    def load(self, split_name):
        if split_name == 'train':
            self.generate_minimal()
            self.split = split_name
            self.train = True
        else:  # in case of validation or test splits

            self.x_split = pd.read_csv(join(self.split_path, split_name + '.csv'), index_col=0)
            self.y_split = np.load(join(self.split_path, split_name + '.npy'))
            self.split = split_name
            self.train = False

    def generate_x_y(self):
        """ Generate the input - output dataset, as we pass the indexes.
        Here we consider as input one of the three splits. If the split from the
        validation or the test set, then we will consider all the samples in it.
        If the split is from the training, we will take into account only the
        number of training samples specific for the experiment.
        In this last case, the training maps are divided, and we consider

        """
        iter_ = self.n_tr[:self.exp.dataset.n_training] if self.split == 'train' \
            else np.arange(self.x_split.shape[0])
        train_y = np.load(join(self.split_path, 'train.npy'))
        # if we keep the original dataset

        if self.exp.dataset.redundant_phrases + self.exp.dataset.noisy_phrases == 0:
            if not self.train:
                return self.x_split.reset_index(drop=True), self.y_split
            else:
                lst_ = [self.x_split.loc[i_, :].values for i_ in iter_]
                return pd.DataFrame(lst_), train_y[iter_]

        # otherwise we load the phrases
        train_fine_phrases = pd.read_csv(join(self.phrases_path,
                                              'fine_phrases_train.csv'), index_col=0)
        # and the original labels
        train_y = np.load(join(self.split_path, 'train.npy'))

        id_phr_n_df = pd.read_csv(join(split_path, 'map_n_%s.csv' % self.split),
                                  index_col=0)
        id_phr_r_df = pd.read_csv(join(split_path, 'map_r_%s.csv' % self.split),
                                  index_col=0)

        for count, id_s in enumerate(iter_):
            # print(id_s)
            stc = [f_ for f_ in self.x_split.loc[id_s] if isinstance(f_, str)]
            id_phr_r = [f_ for f_ in id_phr_r_df.loc[id_s] if not np.isnan(f_)]  # TODO:CHECK
            id_phr_n = [f_ for f_ in id_phr_n_df.loc[id_s] if not np.isnan(f_)]
            id_r = np.random.choice(id_phr_r, size=int(len(id_phr_r) *
                                                   self.exp.dataset.redundant_phrases))
            id_n = np.random.choice(id_phr_n, size=int(len(id_phr_n) *
                                                   self.exp.dataset.noisy_phrases))
            phr_r, phr_n = [], []
            for k_ in id_r:
                phr_r.append([j for j in train_fine_phrases.loc[k_] if isinstance(j, str)])
            for k_ in id_n:
                phr_n.append([j for j in train_fine_phrases.loc[k_] if isinstance(j, str)])

            tmp_lst = stc
            if self.exp.dataset.redundant_phrases > 0:
                tmp_lst = tmp_lst + phr_r
            if self.exp.dataset.noisy_phrases > 0:
                tmp_lst = tmp_lst + phr_n
            tmp_df = pd.DataFrame(np.array(self.flatten(tmp_lst)).reshape(1, -1))

            if count == 0:
                df = tmp_df
            else:
                df = pd.concat([df, tmp_df], axis=0, ignore_index=True)

        return df.reset_index(drop=True), train_y[iter_]