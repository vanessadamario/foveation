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
    dataset_keys = ['dataset_name',
                    'scenario',
                    'additional_dims',
                    'n_training',
                    'redundancy_amount']

    hyper_keys = ['learning_rate',
                  'architecture',
                  'epochs',
                  'batch_size',
                  'loss',
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
                           dataset_name,
                           root_path,
                           result_path,
                           scenario=1,
                           additional_dims=2,
                           architecture='FC',
                           epochs=500,
                           loss='cross_entropy'):
    """ Plot for same scenario and data dimensions.
     Each curve represents the model trained with different amount of data.
     :param df: pandas dataframe from the json, already flatten
     :param dataset_name: string with dataset name
     :param root_path: path where to retrieve results
     :param result_path: path where to store the plots
     :param scenario: int, representing the scenario
     :param additional_dims: int, dimension of the mnist - dataset
     :param architecture: str, architecture type
     :param epochs: int, number of epochs
     :param loss: str, square loss or cross entropy
     """

    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset_name,
                       'additional_dims': additional_dims,
                       'architecture': architecture,
                       'epochs': epochs,
                       'loss': loss}

    n_training_list = np.unique(df['n_training'])
    output_path = join(result_path,
                       dataset_name,
                       architecture,
                       loss,
                       'validation')
    os.makedirs(output_path,
                exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    for n_ in sorted(n_training_list):
        experiment_keys['n_training'] = n_

        # across all batch size and learning rate we pick the best performance
        index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
        best_id = search_best_id(root_path, index_list)

        df_hist = pd.read_csv(join(root_path, 'train_%i/history.csv' % best_id))  # we retrieve id

        bs_ = df.iloc[best_id]['batch_size']  # we plot the convergence time
        n_epochs, _ = df_hist.shape
        convergence_time = (n_epochs * (n_ / bs_)) * np.linspace(0, 1, n_epochs)

        ax[0].loglog(convergence_time, df_hist['loss'], label='t: %i' % n_)
        ax[0].loglog(convergence_time, df_hist['val_loss'], '--', label='v')  # validation loss
        ax[0].set_xlabel('# convergence time', fontsize='xx-large')
        ax[0].set_ylabel('loss', fontsize='xx-large')
        ax[0].legend(fontsize='x-large')
        # ax[0].set_ylim([0, 2])
        ax[1].loglog(convergence_time, df_hist['accuracy'], label='t: %i' %n_)
        ax[1].loglog(convergence_time, df_hist['val_accuracy'], '--', label='v')  # validation accuracy
        ax[1].set_xlabel('# convergence time', fontsize='xx-large')
        ax[1].set_ylabel('accuracy', fontsize='xx-large')
        ax[1].legend(fontsize='x-large')
    fig.suptitle('Validation results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i_dim_%i.pdf' % (scenario,
                                                              additional_dims)))
    plt.close()


def plot_test_curves(df,
                     dataset_name,
                     root_path,
                     result_path,
                     scenario=1,
                     architecture='FC',
                     loss='cross_entropy'):
    """ Plot for same scenario. We change data dimensions and amount of data.
    Here we report the test loss and the test accuracy.

    This plot is for all the additional_dims values

    :param df: flattened dataframe
    :param dataset_name: str
    :param root_path: path where to retrieve results
    :param result_path: path where to save results
    :param scenario: int, denoting the scenario
    :param architecture: str, architecture name
    :param loss: str, name of the loss (cross_entropy or square_loss)
    """

    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset_name,
                       'architecture': architecture,
                       'loss': loss}

    n_training_list = np.unique(df['n_training'])
    dataset_dim_list = np.unique(df['additional_dims'])

    output_path = join(result_path,
                       dataset_name,
                       architecture,
                       loss,
                       'test')
    os.makedirs(output_path,
                exist_ok=True)

    ls_test = np.zeros((len(dataset_dim_list), len(n_training_list)))
    ac_test = np.zeros((len(dataset_dim_list), len(n_training_list)))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)

    for i_, dim_ in enumerate(dataset_dim_list):
        experiment_keys['additional_dims'] = dim_

        for j_, n_ in enumerate(n_training_list):
            experiment_keys['n_training'] = n_
            index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
            # across all batches and learning rate values
            best_id = search_best_id(root_path, index_list)

            ls, ac = np.load(join(root_path, 'train_%i/test.npy' % best_id))
            ls_test[i_, j_] = ls
            ac_test[i_, j_] = ac

        ax[0].loglog(n_training_list, ls_test[i_], 'o-', label='dim %i' % dim_)
        ax[0].set_xlabel('# training samples per class', fontsize='xx-large')
        ax[0].set_ylabel('cross entropy', fontsize='xx-large')

        ax[1].loglog(n_training_list, ac_test[i_], 'o-', label='dim %i' % dim_)
        ax[1].set_xlabel('# training samples per class', fontsize='xx-large')
        ax[1].set_ylabel('accuracy', fontsize='xx-large')

    ax[0].legend(fontsize='x-large')
    ax[1].legend(fontsize='x-large')

    fig.suptitle('Test results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i.pdf' % scenario))
    plt.close()


def plot_epochs_curves(df,
                       dataset_name,
                       root_path,
                       results_path,
                       scenario=1,
                       architecture='FC',
                       epochs=500,
                       loss='cross_entropy'):
    """Here we plot the curves related to the number of epochs before we stop training.

     :param df: flatten pandas DataFrame from the json, already flatten
     :param dataset_name: string with dataset name
     :param root_path: path where to retrieve results
     :param results_path: path where to store the plots
     :param scenario: int, representing the scenario
     :param additional_dims: int, dimension of the mnist - dataset
     :param architecture: str, architecture type
     :param epochs: int, number of epochs
     :param loss: str, square loss or cross entropy

     :returns None: plots
     """
    experiment_keys = {'scenario': scenario,
                       'dataset_name': dataset_name,
                       'architecture': architecture,
                       'epochs': epochs,
                       'loss': loss}

    n_training_array = np.unique(df['n_training'])
    additional_dims_array = np.unique(df['additional_dims'])

    output_path = join(results_path,
                       dataset_name,
                       architecture,
                       loss,
                       'time')
    os.makedirs(output_path,
                exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 5), nrows=1, ncols=3)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    for id_d_, d_ in enumerate(additional_dims_array):
        n_epochs_array = np.zeros(n_training_array.size, dtype=np.int)
        convergence_array = np.zeros(n_training_array.size)
        training_loss_array = np.zeros(n_training_array.size)
        experiment_keys['additional_dims'] = d_

        for id_n_, n_ in enumerate(n_training_array):
            experiment_keys['n_training'] = n_

            # across all batch size and learning rate we pick the best performance
            index_list = generate_bm(df, experiment_keys=experiment_keys)['id'].values
            best_id = search_best_id(root_path, index_list)

            df_hist = pd.read_csv(join(root_path, 'train_%i/history.csv' % best_id))  # we retrieve id

            bs_ = df.iloc[best_id]['batch_size']  # we plot the convergence time
            n_epochs, _ = df_hist.shape
            n_epochs_array[id_n_] = n_epochs
            convergence_array[id_n_] = n_epochs * (n_ / bs_)
            training_loss_array[id_n_] = df_hist['loss'].values[-1]

        ax[0].loglog(n_training_array, n_epochs_array, label='dim %i' %d_)
        ax[1].loglog(n_training_array, convergence_array, label='dim %i' %d_)
        ax[2].loglog(n_training_array, training_loss_array, label='dim %i' %d_)

    ax[0].set_xlabel('# training examples', fontsize='xx-large')
    ax[0].set_ylabel('# epochs', fontsize='xx-large')
    ax[0].legend(fontsize='x-large')

    ax[1].set_xlabel('# training examples', fontsize='xx-large')
    ax[1].set_ylabel('time ', fontsize='xx-large')
    ax[1].legend(fontsize='x-large')

    ax[2].set_xlabel("# training examples ")
    ax[2].set_ylabel("loss at last iteration")
    ax[2].legend(fontsize='x-large')

    fig.suptitle('Convergence time and loss last iteration', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i.pdf' % scenario))
    plt.close()

def plot_test_across_repetitions(root_path_lst,
                                 results_path,
                                 scenario,
                                 architecture,
                                 dataset='standardized_MNIST_dataset',
                                 epochs=500):
    """
    Here we report the test results for the average of multiple repetitions.

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
            ax[0].set_yscale("log", nonposy='clip')
            convergence_time = np.linspace(0, 1, epochs) * epochs * (n_tr_ / bs_)
            ax[0].errorbar(convergence_time,
                           np.nanmean(history_rep, axis=0)[id_n, :, 0],
                           np.nanstd(history_rep, axis=0)[id_n, :, 0],
                           label='t: %i' % n_tr_)
            ax[0].errorbar(convergence_time,
                           np.nanmean(history_rep, axis=0)[id_n, :, 2],
                           np.nanstd(history_rep, axis=0)[id_n, :, 2],
                           label='v', fmt='o')
            ax[0].set_xlabel('# convergence time', fontsize='xx-large')
            ax[0].set_ylabel('cross entropy', fontsize='xx-large')
            ax[0].legend(fontsize='x-large')

            ax[1].errorbar(convergence_time,
                           np.nanmean(history_rep, axis=0)[id_n, :, 1],
                           np.nanstd(history_rep, axis=0)[id_n, :, 1],
                           label='t: %i' % n_tr_)
            ax[1].errorbar(convergence_time,
                           np.nanmean(history_rep, axis=0)[id_n, :, 3],
                           np.nanstd(history_rep, axis=0)[id_n, :, 3],
                           label='v', fmt='o')
            ax[1].set_xlabel('# convergence time', fontsize='xx-large')
            ax[1].set_ylabel('accuracy', fontsize='xx-large')
            ax[1].legend(fontsize='x-large')
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


def plot_pseudoinverse(df,
                       dataset_name,
                       root_path,
                       result_path,
                       scenario=1,
                       architecture='pseudoinverse',
                       loss='square_loss'):
    """ Here we plot the results over the test set for
    the pseudo-inverse solution.
    :param df: the flatten json file
    :param dataset_name: str, name of the dataset
    :param root_path: str, root path of the json to retrieve experiments
    :param result_path: str, where to save the plots
    :param scenario: int, 1,2,4
    :param architecture: str, architecture name
    :param loss: str, name of the loss
    """

    output_path = join(result_path,
                       dataset_name,
                       architecture,
                       'test')
    os.makedirs(output_path,
                exist_ok=True)


    exp_keys = {'dataset_name': dataset_name,
                'scenario': scenario,
                'architecture': architecture,
                'loss': loss}

    # we have different curves for each additional dimensions
    df_pinv = generate_bm(df, exp_keys)
    add_dims_array = np.unique(df_pinv['additional_dims'])
    n_train_array = np.unique(df_pinv['n_training'])

    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    for add_d_ in add_dims_array:
        exp_keys['additional_dims'] = add_d_
        tmp_performance = np.zeros((2, n_train_array.size))

        for id_n_, n_tr_ in enumerate(n_train_array):
            print('index for the exp', id_n_, n_tr_)
            exp_keys['n_training'] = n_tr_
            tmp_output_path = generate_bm(df_pinv, exp_keys)['output_path'].values[0]

            print(join(tmp_output_path, 'test.npy'))

            tmp_performance[:, id_n_] = np.load(join(tmp_output_path, 'test.npy'))

        ax[0].plot(n_train_array, tmp_performance[0], 'o-', label='d:%i' % add_d_)
        ax[1].plot(n_train_array, tmp_performance[1], 'o-', label='d:%i' % add_d_)

    ax[0].set_xlabel("# training examples per class")
    ax[0].set_ylabel("MSE loss")
    ax[0].legend(fontsize='x-large')

    ax[1].set_xlabel("# training examples per class")
    ax[1].set_ylabel("accuracy")
    ax[1].legend(fontsize='x-large')

    fig.suptitle('Test results', fontsize='xx-large')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(join(output_path, 'scenario_%i.pdf' % scenario))


def main():

    repetitions = 1
    dataset_name = 'dataset_2'

    # dataset_name = dataset_1 then repetition_0
    #              = dataset_2 then repetition_1

    if repetitions == 1:
        path_json = '/om/user/vanessad/synthetic_framework/results/repetition_1/train.json'
        path_plots = '/om/user/vanessad/synthetic_framework/plots'
    else:
        lst_path = ['repetition_%i' %i for i in range(repetitions)]
        lst_root_path = [join('/om2/user/vanessad/MNIST_framework/repetitions_500_epochs', res_folder_)
                         for res_folder_ in lst_path]
        path_avg_plot = '/om2/user/vanessad/MNIST_framework/viz_500_epochs_results_avg'

    if repetitions == 1:
        df_ = pd.read_json(path_json)
        df = df_.copy()
        df_flatten = flatten_train_json(df)

        scenario_list = np.unique(df_flatten['scenario'])
        additional_dims = np.unique(df_flatten['additional_dims'])
        architectures = np.unique(df_flatten['architecture'])
        loss_list = np.unique(df_flatten['loss'])

        print('scenarios', scenario_list)
        print('additional dimensions', additional_dims)
        print('architectures', architectures)
        print('loss list', loss_list)

        dct_loss = {'pseudoinverse': ['square_loss'],
                    'linear': ['square_loss', 'cross_entropy'],
                    'FC': ['cross_entropy']}

        for s_ in scenario_list:
            for arch_ in architectures:
                for loss_ in dct_loss[arch_]:
                    if arch_ != 'pseudoinverse':
                        """plot_test_curves(df_flatten,
                                         dataset_name='dataset_2',
                                         root_path=os.path.dirname(path_json),
                                         result_path=path_plots,
                                         scenario=s_,
                                         architecture=arch_,
                                         loss=loss_)"""

                        plot_epochs_curves(df_flatten,
                                           dataset_name=dataset_name,
                                           root_path=os.path.dirname(path_json),
                                           results_path=path_plots,
                                           scenario=s_,
                                           architecture=arch_,
                                           loss=loss_)
                        """for d_ in additional_dims:
                            plot_validation_curves(df_flatten,
                                                   dataset_name='dataset_2',
                                                   root_path=os.path.dirname(path_json),
                                                   result_path=path_plots,
                                                   additional_dims=d_,
                                                   scenario=s_,
                                                   architecture=arch_,
                                                   loss=loss_)"""

                    """else:
                        print('plot for the pseudo-inverse missing!')
                        plot_pseudoinverse(df_flatten,
                                           dataset_name='dataset_2',
                                           root_path=os.path.dirname(path_json),
                                           result_path=path_plots,
                                           scenario=s_,
                                           architecture='pseudoinverse',
                                           loss='square_loss')"""

    """ else:
        for s_ in scenario_list:
            for a_ in architecture_list:
                plot_validation_across_repetitions(lst_root_path, path_avg_plot, s_, a_)
                plot_test_across_repetitions(lst_root_path, path_avg_plot, s_, a_)
                plot_time_across_repetitions(lst_root_path,
                                             path_avg_plot,
                                             scenario=s_,
                                             architecture=a_)
    """


if __name__ == "__main__":
    main()