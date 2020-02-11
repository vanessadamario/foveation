import os
import pandas as pd
import numpy as np
from os.path import join
from search_in_json import search_best_id, flatten_train_json, generate_bm
import matplotlib.pyplot as plt


def plot_validation_curves(df,
                           exp_keys,
                           root_path,
                           result_path):

    """ Plot for same scenario and data dimensions.

            ***WE PLOT THE BEST GENERAL RESULT, ACROSS BATCH SIZE AND LR***

     Each curve represents the model trained with different amount of data.
     :param df: pandas DataFrame from the json, already flatten
     :param exp_keys: dictionary of experiments keys
     :param root_path: str, path where the to look for the train_* folders
     :param result_path: where to save the output
     """
    n_training_lst = np.unique(df['n_training']) # for all possible n_training
    output_path = join(result_path,
                       exp_keys['architecture'],
                       exp_keys['loss'],
                       'validation')
    os.makedirs(output_path, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    for n_ in n_training_lst:
        exp_keys['n_training'] = n_

        index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
        best_id = search_best_id(root_path, index_list)

        df_hist = pd.read_csv(join(root_path,
                                   'train_%i/history.csv' % best_id))  # we retrieve id

        ax[0].plot(df_hist['loss'], label='t: %i' % n_)
        ax[0].plot(df_hist['val_loss'], '--', label='v')  # validation loss
        ax[0].set_xlabel('# epoch', fontsize='xx-large')
        ax[0].set_ylabel('cross entropy', fontsize='xx-large')
        ax[0].legend(fontsize='x-large')
        ax[0].set_ylim([0, 2])
        ax[1].plot(df_hist['accuracy'], label='t: %i' %n_)
        ax[1].plot(df_hist['val_accuracy'], '--', label='v')  # validation accuracy
        ax[1].set_xlabel('# epoch', fontsize='xx-large')
        ax[1].set_ylabel('accuracy', fontsize='xx-large')
        ax[1].legend(fontsize='x-large')
    fig.suptitle('Validation results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i_dim_%i.pdf' % (exp_keys['scenario'],
                                                              exp_keys['additional_dims'])))
    plt.close()


def plot_convergence_time(df,
                          exp_keys,
                          root_path,
                          result_path):

    """ Plot for same scenario and data dimensions.

        *** WE PLOT THE BEST RESULTS ACROSS BATCH SIZE AND LR ***

     Each curve represents the model trained with different amount of data.

     :param df: pandas DataFrame from the json, already flatten
     :param exp_keys: dictionary specifying the parameters for the experiment
     :param root_path: path where to retrieve results
     :param result_path: path where to store the plots
     """

    n_training_lst = np.unique(df['n_training'])
    output_path = join(result_path,
                       exp_keys['architecture'],
                       exp_keys['loss'],
                       'convergence')

    os.makedirs(output_path,
                exist_ok=True)  # we build the validation folder

    tr_lss_curve = []
    vl_lss_curve = []
    tr_acc_curve = []
    vl_acc_curve = []
    convergence_time = []
    number_epochs = []

    for n_ in n_training_lst:
        exp_keys['n_training'] = n_

        index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
        best_id = search_best_id(root_path, index_list)
        bs_ = df.iloc[best_id]['batch_size']
        df_hist = pd.read_csv(join(root_path,
                                   'train_%i/history.csv' % best_id))  # we retrieve id
        n_epochs, _ = df_hist.shape
        convergence_time.append(n_epochs * (n_ / bs_))
        vl_acc_curve.append(df_hist['val_accuracy'].values[-1])
        vl_lss_curve.append(df_hist['val_loss'].values[-1])
        tr_acc_curve.append(df_hist['accuracy'].values[-1])
        tr_lss_curve.append(df_hist['loss'].values[-1])
        number_epochs.append(n_epochs)

    fig, ax = plt.subplots(figsize=(15, 5),
                           nrows=1, ncols=2)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    ax[0].loglog(n_training_lst, tr_lss_curve, 'o-', label='tr')
    ax[0].loglog(n_training_lst, vl_lss_curve, '*-', label='vl')  # validation loss
    ax[0].set_xlabel('#n per class', fontsize='xx-large')
    ax[0].set_ylabel('cross entropy', fontsize='xx-large')
    ax[0].legend(fontsize='x-large')
    ax[1].semilogx(n_training_lst, tr_acc_curve, 'o-', label='tr')
    ax[1].semilogx(n_training_lst, vl_acc_curve, '*-', label='vl')  # validation accuracy
    ax[1].set_xlabel('#n per class', fontsize='xx-large')
    ax[1].set_ylabel('accuracy', fontsize='xx-large')
    ax[1].legend(fontsize='x-large')
    fig.suptitle('Validation results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i_dim_%i.pdf' % (exp_keys['scenario'],
                                                              exp_keys['additional_dims'])))
    plt.close()

    fig, ax = plt.subplots()
    ax.semilogx(n_training_lst, convergence_time, '*-')
    ax.set_xlabel('#n per class', fontsize='xx-large')
    ax.set_ylabel('#epochs', fontsize='xx-large')
    ax.set_title('convergence time', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(join(output_path, 'convergence_%i_dim_%i.pdf' % (exp_keys['scenario'],
                                                                 exp_keys['additional_dims'])))
    plt.close()

    fig, ax = plt.subplots()
    ax.semilogx(n_training_lst, number_epochs, '*-')
    ax.set_xlabel('#n per class', fontsize='xx-large')
    ax.set_ylabel('#epochs', fontsize='xx-large')
    ax.set_title('convergence time', fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(join(output_path, 'epochs_%i_dim_%i.pdf' % (exp_keys['scenario'],
                                                            exp_keys['additional_dims'])))
    plt.close()


def plot_convergence_time_per_batch(df,
                                    exp_keys,
                                    root_path,
                                    result_path):
    """ Fixed scenario.
    Fixed input dimensions.

    Here we report
                    ---- for fixed n ----
                    x axis, batch size
                    y axis, convergence time

    :param df: flattened DataFrame,
    :param exp_keys: dictionary specifying the experiment
    :param root_path: path where to retrieve results
    :param result_path: path where to save results
    """

    n_training_lst = np.unique(df['n_training'])

    output_path = join(result_path,
                       exp_keys['architecture'],
                       exp_keys['loss'],
                       'convergence')
    os.makedirs(output_path,
                exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 6), nrows=1, ncols=4)
    for n_tr in n_training_lst:
        exp_keys['n_training'] = n_tr
        index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
        df_ = df.iloc[index_list]
        batch_size_per_n = np.unique(df_['batch_size'])

        convergence_time = []
        tr_ls = []
        vl_ls = []
        tr_ac = []
        vl_ac = []
        lr = []

        for bs_ in batch_size_per_n:
            exp_keys['batch_size'] = bs_
            index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
            # this will be one of the learning rates
            best_id = search_best_id(root_path, index_list)

            df_hist = pd.read_csv(join(root_path,
                                       'train_%i/history.csv' % best_id))  # we retrieve id
            tr_ls.append(df_hist['loss'].values[-1])
            vl_ls.append(df_hist['val_loss'].values[-1])
            tr_ac.append(df_hist['accuracy'].values[-1])
            vl_ac.append(df_hist['val_accuracy'].values[-1])
            lr.append(df_hist['lr'].values[0])
            n_epochs, _ = df_hist.shape
            convergence_time.append(n_epochs)
        del exp_keys['batch_size']

        ax[0].plot(batch_size_per_n, convergence_time, 'o-', label='n_tr %i' % n_tr)

        ax[1].semilogy(batch_size_per_n, lr, 'o-', label='n_tr %i' % n_tr)

        ax[2].semilogy(batch_size_per_n, tr_ls, 'o-', label='n_tr %i' % n_tr)
        ax[2].semilogy(batch_size_per_n, vl_ls, '--', label='vl')

        ax[3].semilogy(batch_size_per_n, tr_ac, 'o-', label='n_tr %i' % n_tr)
        ax[3].semilogy(batch_size_per_n, vl_ac, '--', label='vl')

    ax[0].set_xlabel('batch size', fontsize='xx-large')
    ax[0].set_ylabel('#epochs', fontsize='xx-large')
    ax[0].legend(fontsize='xx-large')

    ax[1].set_xlabel('batch size', fontsize='xx-large')
    ax[1].set_ylabel('learning rate', fontsize='xx-large')
    ax[1].legend(fontsize='xx-large')

    ax[2].set_xlabel('batch size', fontsize='xx-large')
    ax[2].set_ylabel('loss', fontsize='xx-large')
    ax[2].legend(fontsize='xx-large')

    ax[3].set_xlabel('batch size', fontsize='xx-large')
    ax[3].set_ylabel('accuracy', fontsize='xx-large')
    ax[3].legend(fontsize='xx-large')

    fig.suptitle('Fixed n', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'convergence_fixed_n_%i_dim_%i.pdf' % (exp_keys['scenario'],
                                                                         exp_keys['additional_dims'])))
    plt.close()


def plot_performance_time_fixed_batch(df,
                                      exp_keys,
                                      root_path,
                                      result_path):
    """
    Fixed scenario.
    We save different plot for each dimensions and amount of data.
    Each plot contains different curves, each related to a different batch size.

    We refer to the paper
        *** An Empirical Model of Large-Batch Training ***
            https://arxiv.org/pdf/1812.06162.pdf
    Here we report the test loss and the test accuracy.
                    ---- for fixed n ----
                    x axis, learning rate
                    y axis, validation accuracy

    :param df: flattened DataFrame,
    :param exp_keys: dictionary of paramters
    :param root_path: path where to retrieve results
    :param result_path: path where to save results
    """

    n_training_list = np.unique(df['n_training'])  # for all possible n_training

    output_path = join(result_path,
                       exp_keys['architecture'],
                       exp_keys['loss'],
                       'fixed_batch')
    os.makedirs(output_path,
                exist_ok=True)  # we build the validation folder

    for n_tr in n_training_list:
        exp_keys['n_training'] = n_tr  # for each n
        index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
        df_ = df.iloc[index_list]
        batch_size_per_n = np.unique(df_['batch_size'])

        fig, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)

        for bs_ in batch_size_per_n:
            exp_keys['batch_size'] = bs_
            index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
            df_ = df.iloc[index_list]
            learning_rate_list = np.unique(df_['learning_rate'])

            val_acc = []
            for lr_ in learning_rate_list:
                exp_keys['learning_rate'] = lr_
                # there is just one now
                best_id = generate_bm(df, experiment_keys=exp_keys)['id'].values
                df_hist = pd.read_csv(join(root_path, 'train_%i/history.csv' % best_id))  # we retrieve id
                val_acc.append(df_hist['val_accuracy'].values[-1])

            del exp_keys['learning_rate']
            del exp_keys['batch_size']

            ax.semilogx(learning_rate_list, val_acc, 'o-', label='batch size %i' % bs_)
        ax.set_ylim([0.1, 1.05])
        ax.set_xlabel('learning rate', fontsize='xx-large')
        ax.set_ylabel('validation accuracy', fontsize='xx-large')
        plt.title('Fixed batch, n = %i per class' % n_tr, fontsize='xx-large')
        ax.legend(fontsize='xx-large')
        plt.tight_layout()
        plt.savefig(join(output_path, 'scenario_%i_dim_%i_ntr_%i.pdf' % (exp_keys['scenario'],
                                                                         exp_keys['additional_dims'],
                                                                         exp_keys['n_training'])))


def plot_test_curves(df,
                     exp_keys,
                     root_path,
                     result_path):
    """ Plot for same scenario. We change data dimensions and amount of data.
    Here we report the test loss and the test accuracy.
    :param df: flattened DataFrame,
    :param exp_keys: dictionary containing the main keywords for the experiment
    :param root_path: path where to retrieve results
    :param result_path: path where to save results
    """

    dataset_dim_lst = np.unique(df['additional_dims'])
    n_training_lst = np.unique(df['n_training'])

    output_path = join(result_path,
                       exp_keys['architecture'],
                       exp_keys['loss'],
                       'test')
    os.makedirs(output_path,
                exist_ok=True)

    ls_test = np.zeros((dataset_dim_lst.size, n_training_lst.size))
    ac_test = np.zeros((dataset_dim_lst.size, n_training_lst.size))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)

    for i_, dim_ in enumerate(dataset_dim_lst):
        exp_keys['additional_dims'] = dim_

        for j_, n_ in enumerate(n_training_lst):
            exp_keys['n_training'] = n_

            index_list = generate_bm(df, experiment_keys=exp_keys)['id'].values
            best_id = search_best_id(root_path, index_list)

            ls, ac = np.load(join(root_path, 'train_%i/test.npy' % best_id))
            ls_test[i_, j_] = ls
            ac_test[i_, j_] = ac

        ax[0].loglog(n_training_lst, ls_test[i_], label='dim %i' % dim_)
        ax[0].set_xlabel('# training samples per class', fontsize='xx-large')
        ax[0].set_ylabel('cross entropy', fontsize='xx-large')

        ax[1].semilogx(n_training_lst, ac_test[i_], label='dim %i' % dim_)
        ax[1].set_xlabel('# training samples per class', fontsize='xx-large')
        ax[1].set_ylabel('accuracy', fontsize='xx-large')
    plt.legend(fontsize='x-large')
    fig.suptitle('Test results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i.pdf' % exp_keys['scenario']))
    plt.show()
    plt.close()


def plot_test_across_repetitions(root_path_lst, results_path, scenario, architecture,
                                 dataset='standardized_MNIST_dataset',
                                 epochs=50):
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
                       'architecture': architecture}
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


def plot_validation_across_repetitions(root_path_lst, results_path, scenario, architecture,
                                       dataset='standardized_MNIST_dataset',
                                       epochs=50):
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
                       'architecture': architecture}
    output_path = join(results_path, dataset, architecture, 'train')
    os.makedirs(output_path, exist_ok=True)

    df = flatten_train_json(pd.read_json(join(root_path_lst[0],
                                              'train.json')))
    # these two array must be equivalent to all the repetitions
    # we use the previous scripts to generate clones for the same experiments
    n_training_list = sorted(list(set(df['n_training'])))
    dataset_dim_list = sorted(list(set(df['dataset_dimensions'])))
    epochs = sorted(list(set(df['epochs'])))[-1]  # the max epoch to realize the plots

    for id_d, dim_ in enumerate(dataset_dim_list):
        experiment_keys['dataset_dimensions'] = dim_

        history_rep = np.zeros((len(root_path_lst), len(n_training_list), epochs, 4))
        history_rep[:, :, :, :] = np.nan

        fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)

        for id_n, n_tr_ in enumerate(n_training_list):
            experiment_keys['n_training'] = n_tr_

            experiment_keys['batch_size'] = 10 if n_tr_ < 10 else 32

            for id_rep, root_path_ in enumerate(root_path_lst):
                # print('folder: ', root_path_)
                df = flatten_train_json(pd.read_json(join(root_path_,
                                                          'train.json')))
                index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
                # print(index_list)
                best_id = search_best_id(root_path_, index_list)
                # print('best id', best_id)

                df_hist = pd.read_csv(join(root_path_, 'train_%i/history.csv' % best_id))
                max_size = df_hist.shape[0]
                # (tr_loss, tr_acc, vl_loss, vl_acc)
                # first and last index contain (df.index and learning rate)
                history_rep[id_rep, id_n, :max_size, :] = df_hist.values[:, 1:-1]
            ax[0].set_yscale("log", nonposy='clip')
            ax[0].errorbar(np.arange(epochs),
                           np.nanmean(history_rep, axis=0)[id_n, :, 0],
                           np.nanstd(history_rep, axis=0)[id_n, :, 0],
                           label='t: %i' % n_tr_)
            ax[0].errorbar(np.arange(epochs),
                           np.nanmean(history_rep, axis=0)[id_n, :, 2],
                           np.nanstd(history_rep, axis=0)[id_n, :, 2],
                           label='v', fmt='o')
            ax[0].set_xlabel('# epoch', fontsize='xx-large')
            ax[0].set_ylabel('cross entropy', fontsize='xx-large')
            ax[0].legend(fontsize='x-large')

            ax[1].errorbar(np.arange(epochs),
                           np.nanmean(history_rep, axis=0)[id_n, :, 1],
                           np.nanstd(history_rep, axis=0)[id_n, :, 1],
                           label='t: %i' % n_tr_)
            ax[1].errorbar(np.arange(epochs),
                           np.nanmean(history_rep, axis=0)[id_n, :, 3],
                           np.nanstd(history_rep, axis=0)[id_n, :, 3],
                           label='v', fmt='o')
            ax[1].set_xlabel('# epoch', fontsize='xx-large')
            ax[1].set_ylabel('accuracy', fontsize='xx-large')
            ax[1].legend(fontsize='x-large')
        fig.suptitle('Learning results', fontsize='xx-large')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(join(output_path, 'scenario_%i_dim_%i.pdf' % (scenario,
                                                                  dim_)))
        plt.show()
        plt.close()


def main():

    path_json = '/om/user/vanessad/synthetic_framework/results/repetition_0/train.json'
    path_plots = '/om/user/vanessad/synthetic_framework/visualization'
    os.makedirs(path_plots, exist_ok=True)

    df_ = pd.read_json(path_json)
    df = df_.copy()
    df_flatten = flatten_train_json(df)

    arch_name = 'FC'
    lss = 'cross_entropy'
    experiment_keys = {'architecture': arch_name,
                       'loss': lss}
    scenario_lst = np.unique(df_flatten['scenario'])
    add_dims_lst = np.unique(df_flatten['additional_dims'])
    # n_training_lst = np.unique(df_flatten['n_training'])

    for s_ in scenario_lst:
        for d_ in add_dims_lst:
            experiment_keys['scenario'] = s_
            experiment_keys['additional_dims'] = d_

            plot_validation_curves(df_flatten,
                                   experiment_keys,
                                   root_path=os.path.dirname(path_json),
                                   result_path=path_plots)

            plot_convergence_time(df_flatten,
                                  experiment_keys,
                                  root_path=os.path.dirname(path_json),
                                  result_path=path_plots)

            plot_convergence_time_per_batch(df_flatten,
                                            experiment_keys,
                                            root_path=os.path.dirname(path_json),
                                            result_path=path_plots)

            plot_performance_time_fixed_batch(df_flatten,
                                              experiment_keys,
                                              root_path=os.path.dirname(path_json),
                                              result_path=path_plots)

        plot_test_curves(df_flatten,
                         experiment_keys,
                         root_path=os.path.dirname(path_json),
                         result_path=path_plots)


if __name__ == "__main__":
    main()