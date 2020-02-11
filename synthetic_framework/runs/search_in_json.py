import os
import numpy as np
import pandas as pd
from os.path import join


def search_best_id(root_path, index_list):
    """ Given the indices in the list,
     this function looks for the best realization
    of the experiment, among different learning rates.
    :param root_path: str of the folder containing results
    :param index_list: list of indices for the experiments with same parameters
    """
    loss_val = np.array([pd.read_csv(join(root_path,
                                          'train_%i/history.csv' % index))['val_loss'].values[-1]
                         for index in index_list])  # validation loss at last iteration
    loss_val[np.isnan(loss_val)] = np.inf
    best_id = index_list[np.argmin(loss_val)]       # best index
    return best_id


def flatten_train_json(df):
    """ We assume here that the train.json file has always the same keys.
    :param df: pandas DataFrame
    :return: flatten df, each row corresponds to an experiment.
    """
    df = df.T
    dataset_keys = ['scenario', 'original_dims', 'output_dims',
                    'additional_dims', 'mean_val', 'std_val',
                    'noise', 'noise_mean', 'noise_sigma',
                    'n_training', 'redundancy_amount']
    hyper_keys = ['learning_rate', 'architecture', 'epochs',
                  'batch_size', 'loss', 'optimizer',
                  'lr_at_plateau', 'reduction_factor',
                  'validation_check']
    upper_keys = ['id', 'output_path', 'train_completed']
    columns_name = dataset_keys + hyper_keys + upper_keys

    list_all_samples = []

    for i_ in range(df.shape[0]):
        list_per_sample = []
        for d_k_ in dataset_keys:
            list_per_sample.append(df['dataset'][i_][d_k_])
        for h_k_ in hyper_keys:
            list_per_sample.append(df['hyper'][i_][h_k_])
        for u_p_ in upper_keys:
            list_per_sample.append(df[u_p_][i_])
        list_all_samples.append(list_per_sample)

    return pd.DataFrame(list_all_samples, columns=columns_name)


def generate_bm(df, experiment_keys):
    """ Given the flatten DataFrame, containing all the experiment
    here we extract the experiments correspondent to the dictionary experiment_keys.
    :param df: pandas DataFrame containing the flatten df
    :param experiment_keys: dictionary with all the keys.
    :returns df_copy: the reduced dictionary.
    """
    df_copy = df.copy()
    for (k_, v_) in experiment_keys.items():
        df_copy = df_copy[df_copy[k_] == v_]
    return df_copy


def retrieve_exp_from_json(json_path, experiment_keys):
    """ We collapse the first three function in a unique one.
    :param json_path: the file path to the json file
    :param experiment_keys: the dictionary containing the keys for the experiment.
    :returns data_:
    """
    df = flatten_train_json(pd.read_json(json_path))
    dirname = os.path.dirname(json_path)
    index_list = generate_bm(df, experiment_keys)['id'].values
    df_ = df.iloc[index_list]
    n_exps, keys = df_.shape

    bm = np.array(np.array([len(set(df_[f_]))
                            for f_ in df_.columns
                            if not (f_ == 'id' or f_ == 'output_path')]) != 1)
    if np.sum(bm) > 1:
        raise ValueError('There is more than one free parameter')

    if np.sum(df_['train_completed']) < n_exps:
        raise ValueError('Not all the experiments have been trained')

    best_id = search_best_id(dirname, index_list)
    path_best_id = join(dirname, 'train_%i' % best_id)
    activations = np.load(join(path_best_id, 'activations.npy'))
    history = pd.read_csv(join(path_best_id, 'history.csv'))
    test = np.load(join(path_best_id, 'test.npy'))
    df_ = df.iloc[best_id]

    return [df_, history, activations, test]