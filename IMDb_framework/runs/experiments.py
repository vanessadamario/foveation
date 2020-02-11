import os
import json
import numpy as np
from os.path import join

# if you want to include more architecture, add more elements to the dictionary
# pseudo-inverse, linear with val criterion, one ReLU with val criterion.
transf_dct = {'remove_words': [0., 0.15, 0.30, 0.50, 0.80],
              'change_start': [1, 10, 20, 30, 40]}

n_lst = [5, 20, 50, 100, 1000]
lr_lst = [1, 1e-1, 1e-2, 1e-3, 1e-4]
batch_size_lst = [2, 10, 32, 50]
n_epochs = 500
optimizer_type = 'SGD'
loss = ['cross_entropy']
architectures_hyper = {'linear': {'nodes': [128],
                                  'window': [None]},
                       'FC': {'nodes': [128],
                              'window': [None]},
                       'CNN': {'nodes': [10],  # number of kernels
                               'window': [8]}}


class Hyperparameters(object):
    """ Add hyper-parameters in init so when you read a json, it will get updated as your latest code. """
    def __init__(self,
                 learning_rate=5e-2,
                 architecture=None,
                 nodes=128,
                 window=None,
                 epochs=500,
                 batch_size=10,
                 loss='cross_entropy',
                 optimizer='sgd',
                 lr_at_plateau=True,
                 reduction_factor=None,
                 validation_check=True):
        """

        ** The implemented network is always shallow **

        The hyper-parameter related to the network are specific for the hidden layer

        :param learning_rate: float, the initial value for the learning rate
        :param architecture: str, the architecture types
        :param nodes: int, number of nodes in the architecture. If we use a CNN,
        this is equivalent to the number of filters
        :param window: int or None, specify only for CNN architectures
        :param epochs: int, the number of epochs we want to train
        :param batch_size: int, the dimension of the batch size
        :param loss: str, loss type, cross entropy or square loss
        :param optimizer: str, the optimizer type.
        :param lr_at_plateau: bool, protocol to decrease the learning rate.
        :param reduction_factor, int, the factor which we use to reduce the learning rate.
        :param validation_check: bool, if we want to keep track of validation loss as a stopping criterion.
        """
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.nodes = nodes
        self.window = window
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.lr_at_plateau = lr_at_plateau
        self.reduction_factor = reduction_factor
        self.validation_check = validation_check


class Dataset:
    """ Here we save the dataset specific related to each experiment. The name of the dataset,
    the scenario, if we modify the original dataset, and the dimensions of the input.
    This is valid for the modified_MNIST_dataset, verify if it is going to be valid next"""
    def __init__(self,
                 removed_words=0,
                 first_index=0,
                 n_training=10,
                 embedding_dim=100,
                 output_dims=2):
        """
        :param removed_words: float, percentage of removed words
        :param first_index: int, all the more frequent words are removed
        :param n_training: int, number of training examples
        :param embedding_dim: int, GloVe 100, word2vec 300
        :param output_dims: int, number of classes, two in sentiment analysis
        """
        self.removed_words = removed_words
        self.first_index = first_index
        self.n_training = n_training
        self.embedding_dim = embedding_dim
        self.output_dims = output_dims


class Experiment(object):
    """
    This class represents your experiment.
    It includes all the classes above and some general
    information about the experiment index.
    """
    def __init__(self,
                 id,
                 output_path,
                 train_completed=False,
                 hyper=None,
                 dataset=None):
        """
        :param id: index of output data folder
        :param output_path: output directory
        :param train_completed: bool, it indicates if the experiment has already been trained
        :param hyper: instance of Hyperparameters class
        :param dataset: instance of Dataset class
        """
        if hyper is None:
            hyper = Hyperparameters()
        if dataset is None:
            dataset = Dataset()

        self.id = id
        self.output_path = output_path
        self.train_completed = train_completed
        self.hyper = hyper
        self.dataset = dataset


def exp_exists(exp, info):
    """ This function checks if the experiment
    exists in your json file to avoid duplicate experiments.
    We check if the experiment is already there
    two lists are equivalent if all the elements are equivalent
    """
    # TODO: is this function called in other parts, except from generate_experiments?
    dict_new = exp.__dict__
    dict_new_wo_id = {i: dict_new[i]
                      for i in dict_new if (i != 'id' and i != 'output_path' and i != 'train_completed')}
    for idx in info:
        dict_old = info[idx]
        dict_old_wo_id = {i: dict_old[i]
                          for i in dict_old if (i != 'id' and i != 'output_path' and i != 'train_completed')}
        if dict_old_wo_id == dict_new_wo_id:
            return idx
    return False


def generate_experiments(output_path):
    """ This function is called to make your train.json file or append to it.
    You should change the loops for your own usage.
    The info variable is a dictionary that first reads the json file if there exists any,
    appends your new experiments to it, and dumps it into the json file again
    """
    info = {}
    # empty dictionary

    info_path = output_path + 'train.json'
    dirname = os.path.dirname(info_path)  # we generate the folder
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        idx_base = 0  # first index
    elif os.path.isfile(info_path):
        with open(info_path) as infile:
            info = json.load(infile)
            if info:  # it is not empty
                idx_base = int(list(info.keys())[-1]) + 1  # concatenate
            else:
                idx_base = 0
    else:
        idx_base = 0

    # These loops indicate your experiments. Change them accordingly.
    for net_ in architectures_hyper.keys():
        for nodes_ in architectures_hyper[net_]['nodes']:
            for window_ in architectures_hyper[net_]['window']:
                for loss_ in loss:
                    for rm_words in transf_dct['remove_words']:
                        for mst_freq in transf_dct['change_start']:
                            for lr_ in lr_lst:
                                for n_ in n_lst:
                                    batch_ = [int(b_) for b_ in batch_size_lst if b_ < n_ * 2]
                                    for bs_ in batch_:
                                        hyper = Hyperparameters(learning_rate=lr_,
                                                                architecture=net_,
                                                                nodes=nodes_,
                                                                window=window_,
                                                                batch_size=bs_,
                                                                epochs=n_epochs,
                                                                optimizer=optimizer_type,
                                                                loss=loss_)
                                        dataset = Dataset(removed_words=rm_words,
                                                          first_index=int(mst_freq),
                                                          n_training=n_)
                                        exp = Experiment(id=idx_base,
                                                         output_path=output_path+'train_'+str(idx_base),
                                                         train_completed=False,
                                                         hyper=hyper.__dict__,
                                                         dataset=dataset.__dict__)
                                        print(exp.__dict__)
                                        idx = exp_exists(exp, info)

                                        if idx is not False:
                                            print("The experiment already exists with id:", idx)
                                            continue
                                        info[str(idx_base)] = exp.__dict__
                                        idx_base += 1

    with open(info_path, 'w') as outfile:
        json.dump(info, outfile, indent=4)


def decode_exp(dct):
    """ When reading a json file, it is originally a dictionary
    which is hard to work with in other parts of the code.
    IF YOU ADD ANOTHER CLASS TO EXPERIMENT, MAKE SURE TO INCLUDE IT HERE.
    This function goes through the dictionary and turns it into an instance of Experiment class.
        :parameter dct: dictionary of parameters as saved in the *.json file.
        :returns exp: instance of the Experiment class.
    """
    hyper = Hyperparameters()
    for key in hyper.__dict__.keys():
        if key in dct['hyper'].keys():
            hyper.__setattr__(key, dct['hyper'][key])
    dataset = Dataset()
    for key in dataset.__dict__.keys():
        if key in dct['dataset'].keys():
            dataset.__setattr__(key, dct['dataset'][key])

    exp = Experiment(dct['id'], dct['output_path'], dct['train_completed'], hyper, dataset)
    return exp


def get_experiment(output_path, id):
    """
    This function is called when you want to get the details of your experiment
    given the index (id) and the path to train.json
    """
    info_path = join(output_path, 'train.json')
    with open(info_path) as infile:
        trained = json.load(infile)
    opt = trained[str(id)]  # access to the experiment details through the ID
    exp = decode_exp(opt)   # return an Experiment object

    print('Retrieved experiment:')
    for key in exp.__dict__.keys():
        if key is 'hyper':  # hyper-parameters details
            print('hyper:', exp.hyper.__dict__)
        elif key is 'dataset':  # dataset details
            print('dataset: ', exp.dataset.__dict__)
        else:
            print(key, ':', exp.__getattribute__(key))

    return exp