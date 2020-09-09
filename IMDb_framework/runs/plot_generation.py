import os
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt


def search_best_id(root_path, index_list):
    """ This function, given the parameters, looks for the best realization
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
    :param df: pandas dataframe
    :return: flatten df, each row corresponds to an experiment.
    """
    df = df.T
    dataset_keys = ['redundant_phrases',
                    'noisy_phrases',
                    'n_training',
                    'embedding',
                    'output_dims',
                    'initialization']
    hyper_keys = ['learning_rate',
                  'architecture',
                  'nodes',
                  'window',
                  'epochs',
                  'batch_size',
                  'optimizer',
                  'lr_at_plateau',
                  'reduction_factor',
                  'validation_check']
    upper_keys = ['id',
                  'output_path',
                  'train_completed']

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


def plot_validation_curves(df,
                           root_path,
                           result_path,
                           exp_keys):
    """ Plot for same scenario and data dimensions.
     Each curve represents the model trained with different amount of data.
     :param df: pandas dataframe from the json, already flatten
     :param root_path: path where to retrieve results
     :param result_path: path where to store the plots
     :param exp_keys: dictionary containing the parameters for the exp
     """

    # exp_keys = {'scenario': scenario,
    #                    'dataset_name': dataset,
    #                    'dataset_dimensions': dataset_dimension,
    #                    'architecture': architecture,
    #                    'epochs': epochs}

    n_training_list = np.unique(df['n_training'])
    output_path = join(result_path, exp_keys['architecture'], 'validation')
    os.makedirs(output_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    for n_ in sorted(n_training_list):
        exp_keys['n_training'] = n_

        index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
        best_id = search_best_id(root_path, index_list)
        df_hist = pd.read_csv(join(root_path, 'train_%i/history.csv' % best_id))  # we retrieve id
        bs_ = df.iloc[best_id]['batch_size']
        n_epochs, _ = df_hist.shape
        convergence_time = (n_epochs * (n_ / bs_)) * np.linspace(0, 1, n_epochs)

        ax[0].plot(convergence_time, df_hist['loss'], label='t: %i' % n_)
        ax[0].plot(convergence_time, df_hist['val_loss'], '--', label='v')  # validation loss
        ax[0].set_xlabel('# convergence time', fontsize='xx-large')
        ax[0].set_ylabel('cross entropy', fontsize='xx-large')
        ax[0].legend(fontsize='x-large')
        ax[0].set_ylim([0, 2])
        ax[1].plot(convergence_time, df_hist['accuracy'], label='t: %i' % n_)
        ax[1].plot(convergence_time, df_hist['val_accuracy'], '--', label='v')  # validation accuracy
        ax[1].set_xlabel('# convergence time', fontsize='xx-large')
        ax[1].set_ylabel('accuracy', fontsize='xx-large')
        ax[1].legend(fontsize='x-large')
    fig.suptitle('Validation results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i_dim_%i.pdf' % (exp_keys['scenario'])
    plt.close()


def plot_test_curves(df, root_path, result_path, dataset='standardized_MNIST_dataset',
                     scenario=1, architecture='FC'):
    """ Plot for same scenario. We change data dimensions and amount of data.
    Here we report the test loss and the test accuracy.
    :param df: flattened dataframe,
    :param root_path: path where to retrieve results
    :param result_path: path where to save results
    :param dataset: str, dataset name,
    :param scenario: int, denoting the scenario,
    :param architecture: str, architecture name.
    """

    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset,
                       'architecture': architecture}

    n_training_list = sorted(list(set(df['n_training'])))
    dataset_dim_list = sorted(list(set(df['dataset_dimensions'])))

    output_path = join(result_path, dataset, architecture, 'test')
    os.makedirs(output_path, exist_ok=True)

    ls_test = np.zeros((len(dataset_dim_list), len(n_training_list)))
    ac_test = np.zeros((len(dataset_dim_list), len(n_training_list)))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)

    for i_, dim_ in enumerate(dataset_dim_list):
        experiment_keys['dataset_dimensions'] = dim_

        for j_, n_ in enumerate(n_training_list):
            experiment_keys['n_training'] = n_
            experiment_keys['batch_size'] = 10 if n_ < 10 else 32

            index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
            best_id = search_best_id(root_path, index_list)

            ls, ac = np.load(join(root_path, 'train_%i/test.npy' % best_id))
            ls_test[i_, j_] = ls
            ac_test[i_, j_] = ac

        ax[0].loglog(n_training_list, ls_test[i_], label='dim %i' % dim_)
        ax[0].set_xlabel('# training samples per class', fontsize='xx-large')
        ax[0].set_ylabel('cross entropy', fontsize='xx-large')

        ax[1].loglog(n_training_list, ac_test[i_], label='dim %i' % dim_)
        ax[1].set_xlabel('# training samples per class', fontsize='xx-large')
        ax[1].set_ylabel('accuracy', fontsize='xx-large')
    plt.legend(fontsize='x-large')
    fig.suptitle('Test results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i.pdf' % scenario))
    plt.show()
    plt.close()


def plot_test_across_repetitions(root_path_lst,
                                 results_path,
                                 scenario,
                                 architecture,
                                 dataset='standardized_MNIST_dataset',
                                 epochs=500):
    """ Here we report the test results for the average of multiple repetitions.
    :param root_path_lst: list of str containing all the paths, for each experiment
    :param results_path: str for path where to store the plot
    :param scenario: value correspondent to the scenario
    :param architecture: architecture type
    :param dataset: dataset name, str
    :param epochs: int, number of epochs
    :return None: save plots
    """
    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset,
                       'architecture': architecture,
                       'epochs': epochs}
    output_path = join(results_path, dataset, architecture, 'test')
    os.makedirs(output_path, exist_ok=True)
    flag_find_dim = True
    for id_rep, root_path_ in enumerate(root_path_lst):
        df = flatten_train_json(pd.read_json(join(root_path_,
                                                  'train.json')))
        if flag_find_dim:  # first iteration over all repetitions
            n_training_list = sorted(list(set(df['n_training'])))
            dataset_dim_list = sorted(list(set(df['dataset_dimensions'])))
            ls_test = np.zeros((len(root_path_lst), len(dataset_dim_list), len(n_training_list)))
            ac_test = np.zeros((len(root_path_lst), len(dataset_dim_list), len(n_training_list)))
            flag_find_dim = False
        for id_d, dim_ in enumerate(dataset_dim_list):
            experiment_keys['dataset_dimensions'] = dim_
            for id_n, n_ in enumerate(n_training_list):
                experiment_keys['n_training'] = n_
                if epochs == 50:
                    experiment_keys['batch_size'] = 10 if n_ < 10 else 32
                index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                best_id = search_best_id(root_path_, index_list)
                ls, ac = np.load(join(root_path_, 'train_%i/test.npy' % best_id))
                ls_test[id_rep, id_d, id_n] = ls
                ac_test[id_rep, id_d, id_n] = ac
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].set_xscale("log")
    mn_ls = np.mean(ls_test, axis=0)
    st_ls = np.std(ls_test, axis=0)
    mn_ac = np.mean(ac_test, axis=0)
    st_ac = np.std(ac_test, axis=0)

    for id_d, (dim_, ls_m, ls_s, ac_m, ac_s) in enumerate(zip(dataset_dim_list, mn_ls, st_ls,
                                                mn_ac, st_ac)):
        ax[0].errorbar(n_training_list, ls_m, ls_s, label='dim %i' % dim_)
        ax[1].errorbar(n_training_list, ac_m, ac_s, label='dim %i' % dim_)

    ax[1].set_xlabel('# training samples per class', fontsize='xx-large')
    ax[1].set_ylabel('accuracy', fontsize='xx-large')
    ax[0].set_xlabel('# training samples per class',
                     fontsize='xx-large')
    ax[0].set_ylabel('cross entropy', fontsize='xx-large')
    ax[0].legend(fontsize='x-large')
    ax[1].legend(fontsize='x-large')
    fig.suptitle('Test results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i.pdf' % scenario))
    plt.show()
    plt.close()


def plot_validation_across_repetitions(root_path_lst,
                                       results_path,
                                       scenario,
                                       architecture,
                                       dataset='standardized_MNIST_dataset',
                                       epochs=500):
    """ Here we report the test results for the average of multiple repetitions.
    :param root_path_lst: list of str containing all the paths, for each experiment
    :param results_path: str for path where to store the plot
    :param scenario: value correspondent to the scenario
    :param architecture: architecture type
    :param dataset: dataset name, str
    :param epochs: int, number of epochs
    :return None: save plots
    """
    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset,
                       'architecture': architecture,
                       'epochs': epochs}
    output_path = join(results_path, dataset, architecture, 'train')
    os.makedirs(output_path, exist_ok=True)

    df = flatten_train_json(pd.read_json(join(root_path_lst[0], 'train.json')))
    # these two array must be equivalent to all the repetitions
    # we use the previous scripts to generate clones for the same experiments
    n_training_list = np.sort(np.unique(df['n_training']))
    dataset_dim_list = np.sort(np.unique(df['dataset_dimensions']))
    epochs = sorted(list(set(df['epochs'])))[-1]  # the max epoch to realize the plots

    for id_d, dim_ in enumerate(dataset_dim_list):
        experiment_keys['dataset_dimensions'] = dim_

        history_rep = np.zeros((len(root_path_lst), len(n_training_list), epochs, 4))
        history_rep[:, :, :, :] = np.nan

        # this is because the automatic color code in matplotlib does not exceed ten
        max_color = 5
        n_rows = 2 if n_training_list.size > max_color else 1
        height = 10 if n_training_list.size > max_color else 5

        fig, ax = plt.subplots(figsize=(15, height),
                               squeeze=False,
                               nrows=n_rows,
                               ncols=2)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        for id_n, n_tr_ in enumerate(n_training_list):
            experiment_keys['n_training'] = n_tr_

            id_row = 0
            if n_training_list.size > max_color and id_n > n_training_list.size // 2:
                id_row = 1

            if epochs == 50:
                experiment_keys['batch_size'] = 10 if n_tr_ < 10 else 32

            bs_lst = []
            for id_rep, root_path_ in enumerate(root_path_lst):
                df = flatten_train_json(pd.read_json(join(root_path_,
                                                          'train.json')))
                index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                best_id = search_best_id(root_path_, index_list)
                bs_ = df.iloc[best_id]['batch_size']
                bs_lst.append(bs_)
                df_hist = pd.read_csv(join(root_path_, 'train_%i/history.csv' % best_id))
                max_size, _ = df_hist.shape
                # (tr_loss, tr_acc, vl_loss, vl_acc)
                # first and last index contain (df.index and learning rate)
                history_rep[id_rep, id_n, :max_size, :] = df_hist.values[:, 1:-1]

            bs_ = np.max(bs_lst)
            ax[id_row, 0].set_yscale("log", nonposy='clip')
            convergence_time = epochs * np.linspace(0, 1, epochs) * (n_tr_ / bs_)
            ax[id_row, 0].errorbar(convergence_time,
                                   np.nanmean(history_rep, axis=0)[id_n, :, 0],
                                   np.nanstd(history_rep, axis=0)[id_n, :, 0],
                                   label='t: %i' % n_tr_)
            ax[id_row, 0].errorbar(convergence_time,
                                   np.nanmean(history_rep, axis=0)[id_n, :, 2],
                                   np.nanstd(history_rep, axis=0)[id_n, :, 2],
                                   label='v', fmt='o')
            ax[id_row, 0].set_xlabel('# convergence time', fontsize='xx-large')
            ax[id_row, 0].set_ylabel('cross entropy', fontsize='xx-large')
            ax[id_row, 0].legend(fontsize='x-large')

            ax[id_row, 1].errorbar(convergence_time,
                                   np.nanmean(history_rep, axis=0)[id_n, :, 1],
                                   np.nanstd(history_rep, axis=0)[id_n, :, 1],
                                   label='t: %i' % n_tr_)
            ax[id_row, 1].errorbar(convergence_time,
                                   np.nanmean(history_rep, axis=0)[id_n, :, 3],
                                   np.nanstd(history_rep, axis=0)[id_n, :, 3],
                                   label='v', fmt='o')
            ax[id_row, 1].set_xlabel('# convergence time', fontsize='xx-large')
            ax[id_row, 1].set_ylabel('accuracy', fontsize='xx-large')
            ax[id_row, 1].legend(fontsize='x-large')
        fig.suptitle('Learning results', fontsize='xx-large')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(join(output_path, 'scenario_%i_dim_%i.pdf' % (scenario,
                                                                  dim_)))
        plt.show()
        plt.close()


def plot_time_across_repetitions(root_path_lst,
                                 result_path,
                                 dataset='standardized_MNIST_dataset',
                                 scenario=1,
                                 architecture='FC',
                                 epochs=500):
    """ For each n_tr we report the averaged convergence time
    :param root_path_lst: path where to retrieve results
    :param result_path: path where to store the plots
    :param dataset: string with dataset name
    :param scenario: int, representing the scenario
    :param architecture: str, architecture type
    :param epochs: int, max number of epochs
    """
    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset,
                       'architecture': architecture,
                       'epochs': epochs}
    output_path = join(result_path, dataset, architecture, 'time')
    os.makedirs(output_path, exist_ok=True)

    df = flatten_train_json(pd.read_json(join(root_path_lst[0], 'train.json')))
    n_training_array = np.unique(df['n_training'])
    dataset_dim_array = np.unique(df['dataset_dimensions'])

    convergence_time = np.zeros((dataset_dim_array.size,
                                 n_training_array.size,
                                 len(root_path_lst)))
    n_epochs_across = np.zeros((dataset_dim_array.size,
                                n_training_array.size,
                                len(root_path_lst)),
                               dtype=int)

    for id_r_, root_path_ in enumerate(root_path_lst):
        df = flatten_train_json(pd.read_json(join(root_path_, 'train.json')))

        for id_n_, n_ in enumerate(n_training_array):
            experiment_keys['n_training'] = n_
            if epochs == 50:
                experiment_keys['batch_size'] = 10 if n_ < 10 else 32

            for id_d_, d_ in enumerate(dataset_dim_array):
                experiment_keys['dataset_dimensions'] = d_

                index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                best_id = search_best_id(root_path_, index_list)
                bs_ = df.iloc[best_id]['batch_size']
                df_hist = pd.read_csv(join(root_path_, 'train_%i/history.csv' % best_id))  # we retrieve id
                n_epochs, _ = df_hist.shape
                convergence_time[id_d_, id_n_, id_r_] = n_epochs * (n_ / bs_)
                n_epochs_across[id_d_, id_n_, id_r_] = n_epochs

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    for (d_, ct_per_dim) in zip(dataset_dim_array, convergence_time):
        ct_mean = np.mean(ct_per_dim, axis=-1)
        ct_std = np.std(ct_per_dim, axis=-1)
        ax.errorbar(n_training_array, ct_mean, ct_std, label='d %i' % d_)
    ax.set_xlabel('#n per class', fontsize='xx-large')
    ax.set_ylabel('convergence time', fontsize='xx-large')
    ax.legend(fontsize='x-large')
    plt.tight_layout()
    plt.savefig(join(output_path, 'time_scenario_%i.pdf' % scenario))
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    for (d_, ep_per_dim) in zip(dataset_dim_array, n_epochs_across):
        ep_mean = np.mean(ep_per_dim, axis=-1)
        ep_std = np.std(ep_per_dim, axis=-1)
        ax.errorbar(n_training_array, ep_mean, ep_std, label='d %i' % d_)
    ax.set_xlabel('#n per class', fontsize='xx-large')
    ax.set_ylabel('# epochs', fontsize='xx-large')
    ax.legend(fontsize='x-large')
    plt.tight_layout()
    plt.savefig(join(output_path, 'n_epochs_scenario_%i.pdf' % scenario))
    plt.close()


def auc(x, y):
    """
    Imported from scikit-learn
    Compute Area Under the Curve (AUC) using the trapezoidal rule
    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
    way to summarize a precision-recall curve, see
    :func:`average_precision_score`.
    Parameters
    ----------
    x : array, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array, shape = [n]
        y coordinates.
    Returns
    -------
    auc : float
    0.75
    See also
    --------
    roc_auc_score : Compute the area under the ROC curve
    average_precision_score : Compute average precision from prediction scores
    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds
    """
    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in np.trapz do not return a
        # scalar by default for numpy.memmap instances contrary to
        # regular numpy.ndarray instances.
        area = area.dtype.type(area)
    return area


def plot_auc_across_scenarios(df,
                              root_path_,
                              results_path):
    """ Plot for same scenario. We change data dimensions and amount of data.
    Here we report the test loss and the test accuracy.

    This plot is for all the additional_dims values

    :param root_repetitions_lst: list containing the dataset
    :param root_path: str, path to the folder
    """
    scenario_lst = [1, 2, 4]
    dct_scenario_labels = {1: 'task-uncorrelated',
                           2: 'task-correlated',
                           4: 'mixed'}
    experiment_keys = {}
    n_training_array = np.unique(df['n_training'])
    dim_redundant = np.unique(df['redundant_phrases'])
    dim_noisy = np.unique(df['noisy_phrases'])
    arch_lst = np.unique(df['architecture'])

    for arch_ in arch_lst:
        experiment_keys['architecture'] = arch_
        for id_red, dim_r in enumerate(dim_redundant):
            experiment_keys['redundant_phrases'] = dim_r
            for id_noi, dim_n in enumerate(dim_noisy):
                experiment_keys['noisy_phrases'] = dim_n

                test_accuracy = np.zeros((len(scenario_lst),
                                          dim_redundant.size,
                                          dim_noisy.size,
                                          n_training_array.size))
                for id_s_, s_ in enumerate(scenario_lst):
                    experiment_keys['scenario'] = s_
                    tmp_aucs = np.zeros(dim_redundant.size)
                    for id_n_, n_ in enumerate(n_training_array):
                        experiment_keys['n_training'] = n_
                        index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                        best_id = search_best_id(root_path_, index_list)
                        ls, ac = np.load(join(root_path_, 'train_%i/test.npy' % best_id))
                        test_accuracy[id_s_, id_red, id_noi, id_n_] = ac

                fig, ax = plt.subplots(figsize=(7, 4))
                plt.rc('xtick', labelsize=15)
                plt.rc('ytick', labelsize=15)

                # print(test_accuracy)
                # max_auc = 1 * n_training_array[-1] * 2  # output classes
                for id_s_, s_ in enumerate(scenario_lst):
                    for id_d_, _ in enumerate(dataset_dim_array):
                        tmp_aucs[id_d_] = auc(n_training_array, test_accuracy[id_s_, id_d_, :])
                    if s_ == 4:
                        ax.plot(dataset_dim_array,
                                tmp_aucs[::-1],
                                label=dct_scenario_labels[s_],
                                marker='s',
                                linewidth=3)
                    else:
                        ax.plot(dataset_dim_array,
                                tmp_aucs,
                                label=dct_scenario_labels[s_],
                                marker='s',
                                linewidth=3)
                ax.set_xlabel('image dimensions', fontsize='xx-large')
                ax.set_ylabel('AUC', fontsize='xx-large')
                ax.legend(fontsize='x-large')
                ax.set_ylim([0.5, 1.02])
                plt.tight_layout()
                plt.savefig(join(results_path, 'sst2_AUCs_%s.pdf' % arch_))
                plt.close()

                print(aucs_mean)



def plot_auc_across_scenarios_and_repetitions(root_repetitions_lst,
                                              results_path):
    """ Plot for same scenario. We change data dimensions and amount of data.
    Here we report the test loss and the test accuracy.

    This plot is for all the additional_dims values

    :param root_repetitions_lst: list containing the dataset
    :param root_path: str, path to the folder
    """
    scenario_lst = [1, 2, 4]
    dct_scenario_labels = {1: 'task-unrelated',
                           2: 'task-related',
                           4: 'mixed'}
    experiment_keys = {}
    df = flatten_train_json(pd.read_json(join(root_repetitions_lst[0], 'train.json')))
    n_training_array = np.unique(df['n_training'])
    dataset_dim_array = np.unique(df['dataset_dimensions'])
    arch_lst = np.unique(df['architecture'])

    for arch_ in arch_lst:
        experiment_keys['architecture'] = arch_
        test_accuracy = np.zeros((len(scenario_lst),
                                  dataset_dim_array.size,
                                  n_training_array.size,
                                  len(root_repetitions_lst)))
        aucs_mean = np.zeros((len(scenario_lst),
                              dataset_dim_array.size))
        aucs_std = np.zeros((len(scenario_lst),
                             dataset_dim_array.size))

        for id_r_, root_path_ in enumerate(root_repetitions_lst):
            df = flatten_train_json(pd.read_json(join(root_path_, 'train.json')))

            for id_s_, s_ in enumerate(scenario_lst):
                experiment_keys['scenario'] = s_

                for id_d_, d_ in enumerate(dataset_dim_array):
                    experiment_keys['dataset_dimensions'] = d_

                    for id_n_, n_ in enumerate(n_training_array):
                        experiment_keys['n_training'] = n_

                        index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                        best_id = search_best_id(root_path_, index_list)
                        ls, ac = np.load(join(root_path_, 'train_%i/test.npy' % best_id))
                        test_accuracy[id_s_, id_d_, id_n_, id_r_] = ac

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        # print(test_accuracy)
        # max_auc = 1 * n_training_array[-1] * 2  # output classes
        for id_s_, s_ in enumerate(scenario_lst):
            for id_d_, _ in enumerate(dataset_dim_array):
                tmp_aucs = []
                for id_r_, _ in enumerate(root_repetitions_lst):
                    tmp_aucs.append(auc(n_training_array, test_accuracy[id_s_, id_d_, :, id_r_])/
                                    (n_training_array[-1] - n_training_array[0]))
                aucs_mean[id_s_, id_d_] = np.mean(np.array(tmp_aucs))
                aucs_std[id_s_, id_d_] = np.std(np.array(tmp_aucs))

            if s_ == 4:
                ax.errorbar(dataset_dim_array,
                            aucs_mean[id_s_][::-1],
                            aucs_std[id_s_][::-1],
                            label=dct_scenario_labels[s_],
                            marker='s',
                            linewidth=3)
            else:
                ax.errorbar(dataset_dim_array,
                            aucs_mean[id_s_],
                            aucs_std[id_s_],
                            label=dct_scenario_labels[s_],
                            marker='s',
                            linewidth=3)
        ax.set_xlabel('image dimensions', fontsize='xx-large')
        ax.set_ylabel('AUCs', fontsize='xx-large')
        ax.legend(fontsize='x-large')
        ax.set_ylim([0.5, 1.02])
        plt.tight_layout()
        plt.savefig(join(results_path, 'intemplate_MNIST_AUCs_%s.pdf' % arch_))
        plt.close()

        print(aucs_mean)


def main():

    repetitions = 2

    if repetitions == 1:
        path_json = '/om/user/vanessad/MNIST_framework/results_dynamics/train.json'
        path_plots = '/om/user/vanessad/MNIST_framework/viz_results_dynamics'
    else:
        # lst_path = ['results'] + ['results_%i' % i for i in range(1, repetitions)]
        lst_path = ['repetition_%i' %i for i in range(repetitions)]
        lst_root_path = [join('/om2/user/vanessad/MNIST_framework/repetitions_500_epochs', res_folder_)
                         for res_folder_ in lst_path]
        path_avg_plot = '/om2/user/vanessad/MNIST_framework/viz_500_epochs_results_avg'

    scenario_list = [1, 2, 4]
    data_dim_list = [28, 36, 40, 56, 80, 120, 160]
    architecture_list = ['FC', '2CNN2FC']
    learning_metrics = 4  # loss training validation, acc training validation

    if repetitions == 1:
        df_ = pd.read_json(path_json)
        df = df_.copy()
        df_flatten = flatten_train_json(df)

        for s_ in scenario_list:
            for a_ in architecture_list:
                plot_test_curves(df_flatten, root_path=os.path.dirname(path_json),
                                 result_path=path_plots, scenario=s_, architecture=a_)
                for d_ in data_dim_list:
                    plot_validation_curves(df_flatten,
                                           root_path=os.path.dirname(path_json),
                                           result_path=path_plots,
                                           dataset_dimension=d_,
                                           scenario=s_,
                                           architecture=a_)

    else:

        plot_auc_across_scenarios_and_repetitions(lst_root_path,
                                                  results_path=path_avg_plot)
        # for s_ in scenario_list:
            # for a_ in architecture_list:
                # plot_validation_across_repetitions(lst_root_path, path_avg_plot, s_, a_)
                # plot_test_across_repetitions(lst_root_path, path_avg_plot, s_, a_)
                # plot_time_across_repetitions(lst_root_path,
                #                              path_avg_plot,
                #                              scenario=s_,
                #                              architecture=a_)
                # plot_auc_across_scenarios()


if __name__ == "__main__":
    main()