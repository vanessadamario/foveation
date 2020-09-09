import os
import json
import gensim
import numpy as np
from os.path import join
import pandas as pd
from runs.experiments import Experiment

path_word2vec = '/om/user/vanessad/IMDb_framework/GoogleNews-vectors-negative300.bin'
embed_dct = {'word2vec': {'dimension': 300,
                          'model': gensim.models.KeyedVectors.load_word2vec_format(path_word2vec, binary=True)},
             'glove': {'dimension': 100, 'model': None}
             }


class Embedding:
    """ The class Embedding provides the embedding for a specific embedding, specified in the Experiment instance."""
    def __init__(self):  # , id, exp):
        """ Init for the class Embedding.
        # :param id: identifier of the experiment
        # :param exp: Experiment object
        """
        # self.id = id
        # self.exp = exp

        # self.embed_name = self.exp.dataset.embedding
        # self.init = self.exp.dataset.initialization
        # self.embedding_dim = embed_dct[self.embed_name]['dimension']
        # self.model = embed_dct[self.embed_name]['model']
        # self.min_length_stc = self.exp.hyper.window[-1] \
        #     if isinstance(self.exp.hyper.window, list) else 0
        self.model = embed_dct['word2vec']['model']
        self.min_length_stc = 5
        self.embedding_dim = embed_dct['word2vec']['dimension']
        self.init = 'uniform_all'

        mean_ = np.zeros(self.embedding_dim)
        vocabulary = self.model.vocab.keys()
        len_vocabulary = len(vocabulary)
        for w_ in vocabulary:
            mean_ += self.model[w_]
        mean_ /= len_vocabulary

        sq_ = np.zeros(self.embedding_dim)
        for w_ in vocabulary:
            sq_ += (self.model[w_] - mean_)**2
        var = sq_ / len_vocabulary
        del vocabulary, mean_

        self.var = var
        self.dct_new_words = {}

    def transform_sentence(self,
                           array,
                           fill_dct=True):
        """ Here, given a single sentence, we transform the sentence based on the word2vec embedding, with
        initialization using the uniform distribution in case of not pre-initialized words in the training set.
        We get rid of not pre-initialized words in the validation and test sets.
        :param array: the vector containing the words
        :param fill_dct: bool value, True for sentence in the training set
        :return output_val: it may be -1 if the sentence is not long enough or the array of dimensions
        n_words_in_sentence, 300
        """
        stc = [xx for xx in array if isinstance(xx, str)]
        if len(stc) < self.min_length_stc:
            return -1

        word2vec_stc = []

        for in_word_ in stc:
            if in_word_ not in self.dct_new_words.keys():
                try:
                    word2vec_stc.append(self.model[in_word_])
                except:
                    if fill_dct:
                        tmp_embd = np.array([np.random.uniform(low=-v_, high=v_) for v_ in self.var])
                        self.dct_new_words[in_word_] = list(tmp_embd)
                        word2vec_stc.append(tmp_embd)
            else:
                word2vec_stc.append(np.array(self.dct_new_words[in_word_]))

        if len(word2vec_stc) >= self.min_length_stc:
            return np.array(word2vec_stc)
        else:
            return -1


def embed_dataset(id, source_folder, output_folder):
    """ Here we are at the stage where the three splits starting from the training set have been computed.
    We have then train.csv, valid.csv, test.csv files and the correspondent files with labels.
            [train.csv, valid.csv, test.csv]
            [train.npy, valid.npy, test.npy]
    Then we have three folders, each containing the files map
            [map_valid/map_n_*.csv, map_valid/map_r_*.csv] -- for all the examples
    At this stage we want to obtain the embedding for these files.
    """
    emb = Embedding()
    dct_feat_type = {0: 'n', 1: 'r'}

    fine_phrases = pd.read_csv(join(source_folder, 'fine_phrases_train.csv'), index_col=0)

    for split_ in ['train', 'valid', 'test']:
        path_sentences_ = join(output_folder, split_ + '_sentences')
        path_phrases_ = join(output_folder, split_ + '_phrases')
        os.makedirs(path_sentences_, exist_ok=False)
        os.makedirs(path_phrases_, exist_ok=False)
        x_sentences = pd.read_csv(join(source_folder, split_ + '.csv'), index_col=0)
        y_sentences = np.load(join(source_folder, split_ + '.npy'))

        fill_dct = True if split_ == 'train' else False
        for k_ in range(x_sentences.shape[0]):
            out = emb.transform_sentence(x_sentences.loc[k_], fill_dct)
            if not np.isscalar(out):
                np.save(join(path_sentences_, 'sentence_%i.npy' % k_), out)
                np.save(join(path_sentences_, 'y_%i.npy' % k_), y_sentences[k_])

        fill_dct = False
        for l_ in range(len(os.listdir(join(source_folder, 'map_%s' % split_))) // 2):
            for id_file_, file_map_ in enumerate(['map_n_%i.csv' % l_,
                                                  'map_r_%i.csv' % l_]):

                path_feat_type = join(path_phrases_, 'phrases_%s' % dct_feat_type[id_file_])
                os.makedirs(path_feat_type, exist_ok=True)

                df_phr = pd.read_csv(join(source_folder, 'map_%s' % split_, file_map_), index_col=0)

                for k_ in df_phr.index:
                    id_phrases = [int(id_) for id_ in df_phr.loc[k_] if not np.isnan(id_)]
                    for j_phr, id_phr in enumerate(id_phrases):
                        out = emb.transform_sentence([f_ for f_ in fine_phrases.loc[id_phr]], fill_dct)

                        if not np.isscalar(out):
                            np.save(join(path_feat_type, 'map_%i_%i.npy' % (k_, j_phr)), out)


    def entire_word2vec_variance(self,
                                 input_df,
                                 fill_dct=True):
        """ Based on the variance a of the entire word2vec representation.
        we assign value to the randomly initialized
        words such that U[-a, a]

        :param input_df: pandas data frame containing the sentences
        :param dct_new_words: dictionary containing the new words, not in word2vec
        :param fill_dct: bool, if True we save the new elements
        :returns output_word2vec:
        """
        # dct_path = join(self.exp.output_path, 'dct')
        # os.makedirs(dct_path, exist_ok=True)

        eliminate_samples = False
        eliminated_idx = []

        # if 'dct_not_in_word2vec.json' not in os.listdir(dct_path):
        #    dct_new_words = {}
        # else:
        #     with open(join(dct_path, 'dct_not_in_word2vec.json')) as json_file:
        #         dct_new_words = json.load(json_file)

        output_word2vec = []  # save the representation for the entire set
        for in_sentence_idx in range(input_df.shape[0]):

            word2vec_sentence = []  # save the embedding for the entire sentence here
            out = self.embed_sentence(input_df.loc[in_sentence_idx], fill_dct)
            """
            stc = [f_ for f_ in input_df.loc[in_sentence_idx] if isinstance(f_, str)]

            if len(stc) < self.min_length_stc:
                eliminate_samples = True
                eliminated_idx.append(in_sentence_idx)
                continue

            for in_word_ in stc:  # if the sentence is longer than self.min_length_stc, for each word
                if in_word_ not in dct_new_words.keys():
                    try:  # we try to save the representation of the word
                        word2vec_sentence.append(self.model[in_word_])
                    except:  # otherwise we generate the embedding
                        if fill_dct:
                            rnd_embedding = np.array([np.random.uniform(low=-v_, high=v_) for v_ in self.var])
                            word2vec_sentence.append(rnd_embedding)
                            dct_new_words[in_word_] = list(rnd_embedding)  # save in the dictionary as a list
                else:
                    word2vec_sentence.append(np.array(dct_new_words[in_word_]))  # save in the output embedding as arr
            """

            if len(word2vec_sentence) >= 5:
                output_word2vec.append(np.array(word2vec_sentence))
            else:
                eliminated_idx.append(in_sentence_idx)
        #  with open(join(dct_path, 'dct_not_in_word2vec.json'), 'w') as outfile:
        #     json.dump(dct_new_words, outfile)

        if eliminate_samples:
            return output_word2vec, eliminated_idx, dct_new_words

        return output_word2vec, -1, dct_new_words

    def as_train_word2vec_variance(self,
                                   input_df):
        """ Based on the variance a of the words in the word2vec embedding used in the training set only.
         We assign value to the randomly initialized word such that U[-a, a].

        :param input_df: input data frame

        :returns output_word2vec
        """

        dct_path = join(self.exp.output_path, 'dct')
        os.makedirs(dct_path, exist_ok=True)
        n = input_df.shape[0]
        print(n)
        if 'dct_not_in_word2vec.json' not in os.listdir(dct_path):  # if the dictionary does not exist
            dct_new_words = {}
            word2vec_in_training = []  # words used during training
            not_in_word2vec = []  # words of the training set which are not in the word2vec

            for in_sentence_idx in range(n):  # for all the phrases, we iterate on the words
                stc = [f_ for f_ in input_df.loc[in_sentence_idx] if isinstance(f_, str)]
                print('------------------------')
                print('sentence')
                print(stc)
                if len(stc) < self.min_length_stc:  # if it is too small for the c filter, we continue
                    continue
                print('words in sentence')
                for in_word_ in stc:  # for each word in the sentence
                    print(in_word_)
                    if in_word_ not in not_in_word2vec:
                        # we have not yet stored this word in the dictionary of the unknowns
                        try:
                            # it may be in the word2vec
                            word2vec_in_training.append(self.model[in_word_])
                        except:
                            # if not we need to save it
                            not_in_word2vec.append(in_word_)

            # mean_ = np.zeros(embed_dct[self.embed_name]['dimension'])
            n_words_training = len(word2vec_in_training)  # words from the word2vec
            """for w_ in word2vec_in_training:  # we compute the mean
                mean_ += self.model[w_]"""
            mean_ = np.mean(np.array(word2vec_in_training), axis=0) / n_words_training

            sq_ = np.zeros(embed_dct[self.embed_name]['dimension'])

            for w_ in word2vec_in_training:
                sq_ += (w_ - mean_) ** 2
            var = sq_ / n_words_training  # and the variance

            for k_ in not_in_word2vec:
                dct_new_words[k_] = [np.random.uniform(low=-v_, high=v_) for v_ in var]

            with open(join(dct_path, 'dct_not_in_word2vec.json'), 'w') as outfile:
                json.dump(dct_new_words, outfile)
            np.save(join(dct_path, 'var.npy'), var)

        with open(join(dct_path, 'dct_not_in_word2vec.json')) as json_file:  # repetition for the training
            dct_new_words = json.load(json_file)  # otherwise we load it
        var = np.load(join(dct_path, 'var.npy'))

        # second part, here we generate the representation
        eliminate_samples = False
        eliminated_idx = []

        output_word2vec = []

        for in_sentence_idx in range(n):
            word2vec_sentence = []
            stc = [f_ for f_ in input_df.loc[in_sentence_idx] if isinstance(f_, str)]
            if len(stc) < self.min_length_stc:
                eliminate_samples = True
                eliminated_idx.append(in_sentence_idx)
                continue
            for in_word_ in stc:
                if in_word_ not in dct_new_words.keys():
                    try:
                        word2vec_sentence.append(self.model[in_word_])
                    except:
                        rnd_embedding = np.array([np.random.uniform(low=-v_, high=v_) for v_ in var])
                        word2vec_sentence.append(rnd_embedding)
                else:
                    word2vec_sentence.append(np.array(dct_new_words[in_word_]))  # save in the output embedding as arr

            output_word2vec.append(np.array(word2vec_sentence))

        if eliminate_samples:
            return output_word2vec, eliminated_idx

        return output_word2vec, -1

    def convert_word2vec(self, input_df):
        """ Here we pass the input data. The input data consists of list of lists.
        Each sample corresponds to a list of words. The preprocess is such that we have no space, neither capital letters.
        We give here the representation using the word2vec embedding.
        Each word has a representation of size 300.

        :param input_df: it is a list of lists. Each lists contains a sentence or a phrase.

        :returns: a list of numpy arrays, of size #n_words_in_sentence/phrase, 300
        """

        dct_path = join(self.exp.output_path, 'dct')
        os.makedirs(dct_path, exist_ok=True)

        n = input_df.shape[0]
        eliminate_samples = False
        eliminated_idx = []

        if 'dct_not_in_word2vec.json' not in os.listdir(dct_path):
            dct_new_words = {}
        else:
            with open(join(dct_path, 'dct_not_in_word2vec.json')) as json_file:
                dct_new_words = json.load(json_file)

        output_word2vec = []
        for in_sentence_idx in range(n):
            word2vec_sentence = []
            stc = [f_ for f_ in input_df.loc[in_sentence_idx] if isinstance(f_, str)]
            # print(stc)
            if len(stc) < self.min_length_stc:
                eliminate_samples = True
                eliminated_idx.append(in_sentence_idx)
                continue
            for in_word_ in stc:
                if in_word_ not in dct_new_words.keys():
                    try:
                        word2vec_sentence.append(self.model[in_word_])
                    except:
                        # print("random initialization")
                        rnd_embedding = np.random.randn(embed_dct[self.embed_name]['dimension'])
                        word2vec_sentence.append(rnd_embedding)
                        dct_new_words[in_word_] = list(rnd_embedding)  # save in the dictionary as a list
                else:
                    word2vec_sentence.append(np.array(dct_new_words[in_word_]))  # save in the output embedding as arr

            output_word2vec.append(np.array(word2vec_sentence))

        with open(join(dct_path, 'dct_not_in_word2vec.json'), 'w') as outfile:
            json.dump(dct_new_words, outfile)

        if eliminate_samples:
            return output_word2vec, eliminated_idx

        return output_word2vec, -1

    def apply(self, input_df, y):
        """ We apply one of the previous transformations, depending on the dct initialization
        :param input_df: input data frame
        :param y: output label """

        embed_and_init_unknowns = {'gaussian': self.convert_word2vec,
                                   'uniform_all': self.entire_word2vec_variance,
                                   'uniform_tr': self.as_train_word2vec_variance}
        embedded_lst, eliminated_id = embed_and_init_unknowns[self.init](input_df)
        if eliminated_id != -1:
            y = np.delete(y, eliminated_id)

        if self.exp.hyper.architecture == 'CNN':
            return embedded_lst, y

        else:  # average over all the words
            return np.array([np.mean(emb_, axis=0) for emb_ in embedded_lst]), y

