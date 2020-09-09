import os
import numpy as np
from os.path import join
import re
import codecs
import pandas as pd


def clean_str_sst(string):
    """
    Tokenize/string cleaning for the SST dataset
    :param string: a string element
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def line_to_words(line):
    """ Given the dataset, we remove the first element, which is the label
    :param line: line of the dataset, as saved for e.g. in
    https://github.com/CS287/HW1/blob/master/data/stsa.binary.train
    :returns label: the output label
    :returns words: the list of words contained in the line
    """
    clean_line = clean_str_sst(line.strip())
    words = clean_line.split(' ')
    label = int(words[0])
    words = words[1:]
    return label, words


def extract_x_y(path_folder, dataset_name):
    """ Here we extract the (X,y) values
    :param path_folder: path to the folder containing datasets
    :param dataset_name: name of the dataset
    :returns input_dataset_lst: list containing input features (list of words)
    :returns output_labels: np.array containing the ys
    """
    input_dataset_lst = []
    output_labels = []
    for filename in [join(path_folder, dataset_name)]:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    label, words = line_to_words(line)
                    input_dataset_lst.append(words)
                    output_labels.append(label)

    return input_dataset_lst, np.array(output_labels)


def standard_pre_processing(id, source_folder, split_path):
    output_name = {'stsa.binary.phrases.train': 'phrases_train',
                   'stsa.binary.train': 'train',
                   'stsa.binary.dev': 'valid',
                   'stsa.binary.test': 'test'}

    dataset_lst = list(output_name.keys())

    for data_ in dataset_lst:
        x, y = extract_x_y(source_folder, data_)
        df = pd.DataFrame(x)
        df.to_csv(join(split_path, output_name[data_] + '.csv'))
        np.save(join(split_path, output_name[data_] + '.npy'), y)


def generate_map(id, source_folder, split_path, min_length=5):
    """ Generate the map from sentence to phrases.
    The redundant phrases are those which share the same polarity of the entire sentence.
    The noisy phrases are the neutral ones, with label 2
    :param id: the index of the experiment, not useful
    :param source_folder: name of the source, containing the fine grained phrases
    :param split_path: the name of the split where to save the results
    :param min_length: the minimum length for each sentence, default 5, min length of the sentence
    """
    split_shape = 10
    x_phr = pd.read_csv(join(source_folder, 'fine_phrases_train.csv'), index_col=0)
    y_phr = np.load(join(source_folder, 'fine_phrases_train.npy'))

    split_name_lst = ['train', 'valid', 'test']

    for split_name in split_name_lst:
        dct_redundant = {}  # dct of redundant phrases
        dct_noise = {}      # dct of noisy phrases

        x_split = pd.read_csv(join(split_path, split_name + '.csv'), index_col=0)
        y_split = np.load(join(split_path, split_name + '.npy'))
        size_split = x_split.shape[0]
        n_index_lst = np.ceiling(size_split / split_shape).astype(int)

        for n_index in n_index_lst:
            id_start = n_index * split_shape
            id_stop = (n_index + 1) * split_shape
            if id_stop > size_split:
                id_stop = size_split

            for id_s in np.arange(id_start, id_stop):
                print(id_s)
                ff = [f_ for f_ in x_split.loc[id_s] if isinstance(f_, str)]  # for each sentence in the file_split.csv
                dct_redundant[id_s] = []
                dct_noise[id_s] = []

                for id_phr in range(x_phr.shape[0]):  # you look for all the phrases
                    tmp_phrases = [f_ for f_ in x_phr.loc[id_phr] if isinstance(f_, str)]
                    len_phrases = len(tmp_phrases)
                    set_phrases = set(tmp_phrases)
                    if len(set_phrases - set(ff)) == 0 and len_phrases >= min_length:
                        # save the phrase as the one corresponding to the sentence
                        if ((y_phr[id_phr] > 2) and y_split[id_s] == 1) \
                                or ((y_phr[id_phr] < 2) and y_split[id_s] == 0):
                            dct_redundant[id_s].append(id_phr)

                        elif y_phr[id_phr] == 2:
                            dct_noise[id_s].append(id_phr)

            df_n = pd.DataFrame(dct_noise.values(), index=dct_noise.keys())
            df_n.to_csv(join(split_path, 'map_' + split_name, 'map_n_%i.csv' % n_index))
            df_r = pd.DataFrame(dct_redundant.values(), index=dct_redundant.keys())
            df_r.to_csv(join(split_path, 'map_' + split_name, 'map_r_%i.csv' % n_index))