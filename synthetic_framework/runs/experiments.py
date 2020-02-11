import json
import os
import numpy as np
from os.path import join


# if you want to include more architecture, add more elements to the dictionary
# pseudo-inverse, linear with val criterion, one ReLU with val criterion.


exp_params_dct = {'dataset_5': {'original_dims': 30,
                                'output_dims': 2,
                                'additional_dims': [2, 5, 10, 50],
                                'n_array': [2, 5, 20, 50, 100],
                                'redundancy_amount': [0.5]}
                  }

"""
exp_params_dct = {'dataset_1': {'original_dims': 30,
                                'output_dims': 2,
                                'additional_dims': [2, 5, 10, 50],
                                'n_array': [2, 5, 20, 50, 100],
                                'redundancy_amount': [0.5]},
                  'dataset_2': {'original_dims': 30,
                                'output_dims': 2,
                                'additional_dims': [2, 5, 10, 50],
                                'n_array': [2, 5, 20, 50, 100],
                                'redundancy_amount': [0.5]}
                  'dataset_4': {'original_dims': 30,
                                'output_dims': 2,
                                'additional_dims': [2, 5, 10, 50],
                                'n_array': [2, 5, 20, 50, 100],
                                'redundancy_amount': [0.5]}
                  }"""
# dataset 3 is same as four and so on


lr_array = [1e-2, 1e-3, 1e-5]
scenarios = [1, 2, 4]
batch_size_array = np.array([2, 10, 32, 50], dtype=int)

lr_dct = {'pseudoinverse': [None],
          'linear': lr_array,
          'FC': lr_array}  # we iterate on these elements

epochs_dct = {'pseudoinverse': None,
              'linear': 500,
              'FC': 500}  # we do not iterate on this

loss_dct = {'pseudoinverse': ['square_loss'],
            'linear': ['square_loss', 'cross_entropy'],
            'FC': ['cross_entropy']}  # again iteration

optimizer_dct = {'pseudoinverse': None,
                 'linear': 'sgd',
                 'FC': 'sgd'}

# this is for scenario 1, 4, differently from the others

# we need a rule for the original dimensions
# we need a standard deviation for each feature


class Hyperparameters(object):
    """ Add hyper-parameters in init so when you read a json, it will get updated as your latest code. """
    def __init__(self,
                 learning_rate=5e-2,
                 architecture=None,
                 epochs=500,
                 batch_size=10,
                 loss='cross_entropy',
                 optimizer='sgd',
                 lr_at_plateau=True,
                 reduction_factor=None,
                 validation_check=True):
        """
        :param learning_rate: float, the initial value for the learning rate
        :param architecture: str, the architecture types
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
    # TODO: add output_dims
    def __init__(self,
                 dataset_name='dataset_1',
                 scenario=1,
                 additional_dims=2,
                 n_training=10,
                 redundancy_amount=None):
        """
        :param dataset_name: str, dataset name
        :param scenario: int, the learning paradigm
        :param additional_dims: int, additional noise
        :param n_training: int, number of training examples
        :param redundancy_amount, percentage of redundant features, scenario 4 only
        """
        self.dataset_name = dataset_name
        self.scenario = scenario
        self.additional_dims = additional_dims
        self.n_training = n_training
        self.redundancy_amount = redundancy_amount


class Experiment(object):
    """
    This class represents your experiment.
    It includes all the classes above and some general
    information about the experiment index.
    IF YOU ADD ANOTHER CLASS, MAKE SURE TO INCLUDE IT HERE.
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
    # do we want to put also the flag train_completed here, correct?
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
    info = {}  # empty dictionary

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
    for data_name in exp_params_dct.keys():
        for net_ in lr_dct.keys():
            for loss_ in loss_dct[net_]:
                for spec_scenario_ in scenarios:  # for each element of the dictionary
                    if spec_scenario_ == 4:
                        redundancy_amount_lst = exp_params_dct[data_name]['redundancy_amount']
                    else:
                        redundancy_amount_lst = [None]

                    for add_dims_ in exp_params_dct[data_name]['additional_dims']:
                        for red_amount_ in redundancy_amount_lst:
                            for lr_ in lr_dct[net_]:
                                for n_ in exp_params_dct[data_name]['n_array']:

                                    batch_ = batch_size_array[batch_size_array < n_ *
                                                              exp_params_dct[data_name]['output_dims']]
                                    batch_ = [int(batch_)] if np.isscalar(batch_) else batch_
                                    if net_ == 'pseudoinverse':
                                        batch_ = [None]
                                    for bs_ in batch_:
                                        if bs_ is not None:
                                            bs_ = int(bs_)
                                        hyper = Hyperparameters(learning_rate=lr_,
                                                                architecture=net_,
                                                                batch_size=bs_,
                                                                epochs=epochs_dct[net_],
                                                                optimizer=optimizer_dct[net_],
                                                                loss=loss_)
                                        dataset = Dataset(dataset_name=data_name,
                                                          scenario=spec_scenario_,
                                                          additional_dims=add_dims_,
                                                          n_training=n_,
                                                          redundancy_amount=red_amount_)
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