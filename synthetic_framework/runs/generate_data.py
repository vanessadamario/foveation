import os
import numpy as np
from os.path import join

dataset_spec = {'dataset_5': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(0.5 * np.round(np.random.randn(30), decimals=3)),
                                           list(-0.5 *np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(0.5 * np.ones(30)),
                                          list(0.5 * np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0.,
                              'noise_sigma': 1.,
                              'n_samples_per_class': 5000
                              }
                }


"""dataset_spec = {'dataset_1': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(2 * np.round(np.random.randn(30), decimals=3)),
                                           list(-2 * np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(0.5 * np.ones(30)),
                                          list(0.5 * np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0.,
                              'noise_sigma': 0.5,
                              'n_samples_per_class': 5000
                             },
                 
                'dataset_2': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(2 * np.round(np.random.randn(30), decimals=3)),
                                           list(-2 * np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(np.ones(30)),
                                          list(np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0,
                              'noise_sigma': 0.5,
                              'n_samples_per_class': 5000},
                              
                'dataset_3': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(np.round(np.random.randn(30), decimals=3)),
                                           list(-np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(0.5 * np.ones(30)),
                                          list(0.5 * np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0.,
                              'noise_sigma': 0.5,
                              'n_samples_per_class': 5000
                              },
                'dataset_4': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(0.5 * np.round(np.random.randn(30), decimals=3)),
                                           list(-0.5 *np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(0.5 * np.ones(30)),
                                          list(0.5 * np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0.,
                              'noise_sigma': 0.5,
                              'n_samples_per_class': 5000
                              }
                'dataset_5': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(0.5 * np.round(np.random.randn(30), decimals=3)),
                                           list(-0.5 *np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(0.5 * np.ones(30)),
                                          list(0.5 * np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0.,
                              'noise_sigma': 1.,
                              'n_samples_per_class': 5000
                              }
                'dataset_6': {'original_dims': 30,
                              'output_dims': 2,
                              'max_additional_dims': 50,
                              'mean_val': [list(0.5 * np.round(np.random.randn(30), decimals=3)),
                                           list(-0.5 *np.round(np.random.randn(30), decimals=3))],
                              'std_val': [list(0.5 * np.ones(30)),
                                          list(0.5 * np.ones(30))],
                              'noise': 'gaussian',
                              'noise_mean': 0.,
                              'noise_sigma': 2.,
                              'n_samples_per_class': 5000
                              }
                }"""


class DatasetGenerator:
    """ Class for the data set generation. We generate the necessary components
    to create the dataset, which are training, validation and test (X, y) for
    a classification task. To put additional noise we include a matrix of
    random number for each split and a linear transformation F.
    To create the dataset for a specific experiment we need to call the
    method generate_input_experiment.
    """

    def __init__(self,
                 data_path=None,
                 load=False,
                 key_dataset=None,  # dataset_spec['dataset_1']
                 exp=None,
                 cols='features'):
        """
        Generate dataset for a supervised learning task. The features are extracted using
        Gaussian distributions.
        :param data_path: where to save data or load data from
        :param load: bool, if True we load the data already generated
        :param key_dataset: str, dataset name
        :param exp: object from the Experiment class
        :param cols: str, set to features to have output [n_samples, n_features]
        """
        self.data_path = data_path
        self.load = load
        self.dct_dataset = dataset_spec[key_dataset]
        self.exp = exp
        self.cols = cols

        self.minimal_dataset = False
        self.splits_lst = ['train', 'validation', 'test']
        self.A = None
        self.X_splits = None
        self.y_splits = None
        self.noise_splits = None

        if self.dct_dataset is not None:
            self.p = self.dct_dataset['original_dims']
            self.K = self.dct_dataset['output_dims']
            self.max_add_p = self.dct_dataset['max_additional_dims']
            self.N_per_class = self.dct_dataset['n_samples_per_class']
            self.N = self.K * self.N_per_class
            self.mu_array = np.array(self.dct_dataset['mean_val'])
            self.sigma_array = np.array(self.dct_dataset['std_val'])

            if not self.load:
                self.save_minimal_data()
                self.minimal_data = True

        if load:
            if self.data_path is None:
                raise ValueError("You need to provide a path to the dataset")
            else:
                self.load_minimal_data()
                self.minimal_dataset = True

        if exp is not None:
            self.exp = exp

    def _generate_minimal_data(self):
        """ Here we generate the data by using the relevant features only.
        Each feature is Gaussian distributed. Mean and standard
        deviation for each variable varies depending on the user specification.

        The generic i-th feature is x_i
                    x_i = mean_i + N(0,1) * std_i, x_i in R^n_samples

        The labels are generating depending on the learning task.
        The classifier the two distribution are
        given different values. # at the moment we are not considering the
        multi-classification task.
        """
        check_output_mu, check_input_mu = (np.squeeze(self.mu_array)).shape
        check_output_st, check_input_st = (np.squeeze(self.sigma_array)).shape

        if check_output_mu != self.K or check_output_st != self.K:
            raise ValueError("Arrays inconsistent with the number of classes")

        X_ = np.zeros((self.p, self.N))
        y_ = np.zeros((self.K, self.N))

        for k_, (mu_class_, sigma_class_) in enumerate(zip(self.mu_array,
                                                           self.sigma_array)):  # for each class
            first_ = k_ * self.N_per_class  # n_per_class
            last_ = self.N if k_ == self.K - 1 else (k_ + 1) * self.N_per_class
            for id_, (mu_, sigma_) in enumerate(zip(mu_class_, sigma_class_)):
                X_[id_, first_:last_] = mu_ + np.random.randn(last_ - first_) * sigma_
            y_[k_, first_:last_] = 1

        self.y = y_
        self.X = X_
        self.minimal_dataset = True

        return self

    def load_minimal_data(self):
        """ Here we load the dataset, if already generated. """
        self.A = np.load(join(self.data_path, 'A.npy'))
        X_s, y_s, noise_s = [], [], []
        for fold_ in self.splits_lst:
            X_s.append(np.load(join(self.data_path, fold_, 'X.npy')))
            y_s.append(np.load(join(self.data_path, fold_, 'y.npy')))
            noise_s.append(np.load(join(self.data_path, fold_, 'N.npy')))
        self.X_splits = X_s
        self.y_splits = y_s
        self.noise_splits = noise_s

    def save_minimal_data(self):
        """ Here we save the ingredients to generate any dataset. """
        self.A = np.random.randn(self.max_add_p, self.p)
        np.save(join(self.data_path, 'A.npy'), self.A)

        for id_split_, fold_ in enumerate(self.splits_lst):
            fold_data = join(self.data_path, fold_)
            os.makedirs(fold_data, exist_ok=True)
            self._generate_minimal_data()

            self.noise = np.random.randn(self.max_add_p, self.N)
            np.save(join(self.data_path, fold_, 'X.npy'), self.X)
            np.save(join(self.data_path, fold_, 'y.npy'), self.y)
            np.save(join(self.data_path, fold_, 'N.npy'), self.noise)

    def add_redundancy(self):
        """ We add redundancy to the dataset.
        Using a linear combination of the input features.
        """
        self.A = self.A[:self.exp.dataset.additional_dims, :]
        X_splits_ = []
        for x_data_, f_ in zip(self.X_splits,
                               self.splits_lst):
            X_splits_.append(np.vstack((x_data_,
                                        np.dot(self.A, x_data_))))
        return X_splits_, self.y_splits

    def add_gaussian_noise(self):
        """ We add noisy features to the dataset.
        This is done by adding Gaussian distributed
        random variables to the original features.
        """
        if not self.minimal_dataset:
            raise ValueError("Generate the dataset first")
        X_splits_ = []
        for x_data_, n_data_, f_ in zip(self.X_splits,
                                        self.noise_splits,
                                        self.splits_lst):
            X_splits_.append(np.vstack((x_data_,
                                        n_data_[:self.exp.dataset.additional_dims])))
        return X_splits_, self.y_splits

    def add_mixture(self, n_noise_feat, n_rdndt_feat):
        """ With this call we add a percentage of redundancy and a (1-percentage) of noisy features. """
        self.A = self.A[:n_rdndt_feat, :]  # the first n_rdndt components
        X_splits_ = []  # we have the three splits
        for x_data_, n_data_, f_ in zip(self.X_splits,
                                        self.noise_splits,
                                        self.splits_lst):
            tmp_ = np.vstack((x_data_, n_data_[:n_noise_feat]))
            X_splits_.append(np.vstack((tmp_, np.dot(self.A, x_data_))))

        return X_splits_, self.y_splits

    def _get_n_train_elements_per_class(self):
        """ Consider a fixed amount of training data. """
        if not self.minimal_dataset or self.exp is None:
            raise ValueError("Generate the dataset first")

        y_tr = self.y_splits[0]  # (k, n)
        n_classes, n_samples = y_tr.shape
        n_s_per_class = self.exp.dataset.n_training

        idx = np.array([], dtype=int)
        for k in range(n_classes):
            idx = np.append(idx, np.arange(k * (n_samples // n_classes),
                                           k * (n_samples // n_classes) + n_s_per_class))
        return idx

    def generate_input_experiment(self):
        """ Generate the dataset (X, y) for a specific experiment. """
        idx = self._get_n_train_elements_per_class()
        self.X_splits[0] = self.X_splits[0][:, idx]
        self.y_splits[0] = self.y_splits[0][:, idx]
        self.noise_splits[0] = self.noise_splits[0][:, idx]

        if self.exp.dataset.scenario == 1:
            [X_exp, y_exp] = self.add_gaussian_noise()

        elif self.exp.dataset.scenario == 2:
            [X_exp, y_exp] = self.add_redundancy()

        elif self.exp.dataset.scenario == 4:
            r_ = self.exp.dataset.redundancy_amount
            n_noise_feat = int(np.ceil(self.exp.dataset.additional_dims * (1 - r_)))
            n_rdndt_feat = int(np.floor(self.exp.dataset.additional_dims * r_))
            [X_exp, y_exp] = self.add_mixture(n_noise_feat, n_rdndt_feat)

        else:
            raise ValueError("Scenario does not exists")

        if self.cols == 'features':
            return [[X_.T for X_ in X_exp], [y_.T for y_ in y_exp]]

        return [X_exp, y_exp]

##########################################################
# def save_data(id, output_path):
    """
    Find the hyper-parameters to generate the data.

    We save the data with dimensions (p, n), with p features and n samples.

    :param id: not necessary
    :param output_path: output path where we store the data and from which
    we load the *.json file
    :return: None, it generates the dataset based on what is contained in the json
    file
    

    for k_ in dataset_spec.keys():

        dataset_path = join(output_path, k_)
        os.makedirs(dataset_path, exist_ok=False)

        A = np.random.randn(dataset_spec[k_]['max_additional_dims'],
                            dataset_spec[k_]['original_dims'])
        np.save(join(dataset_path, 'A.npy'), A)

        folder_lst = ['train', 'validation', 'test']

        for id_split_, f_name in enumerate(folder_lst):
            data_gen = DatasetGenerator(p=int(dataset_spec[k_]['original_dims']),
                                        K=int(dataset_spec[k_]['output_dims']),
                                        N_per_class=int(dataset_spec[k_]['n_samples_per_class']))
            data_gen.generate_minimal_data()
            np.save(join(dataset_path, f_name, 'X.npy'), data_gen.X)
            np.save(join(dataset_path, f_name, 'y.npy'), data_gen.y)

            N = np.random.randn(dataset_spec[k_]['max_additional_dims'],
                                int(dataset_spec[k_]['n_samples_per_class']))
            np.save(join(dataset_path, f_name, 'N.npy'), N)



class DatasetGenerator:
     Class for the data set generation. We consider three scenarios = [1,2,4].
    Each related to a different transformation of the low dimensional data.
    We generate DatasetGenerator objects everytime we generate a model and create
    a sample split.

    In the case of redundant transformation
        ** pass the linear transformation as a dct_kwargs['A'] argument **

    The risk otherwise is to have three different linear transformation for the
    training, validation, and test dataset splits.
    Given the input argument, the class initialization already generate the input
    output relations, with the transformations of interest.

    If the noise mean and standard deviations are not specified and we are in scenario
    1 or 4, we generate normally distributed features.

    
    def __init__(self,
                 p,
                 N_per_class,
                 K,
                 class_task=False,
                 scenario=1,
                 mu_array=None,
                 sigma_array=None,
                 **kwargs):
        
        Generate dataset for a supervised learning task. The features are extracted using
        Gaussian distributions.
        :param p: int, of relevant features
        :param N_per_class: int, number of training data per class
        :param K: int, output dimension for regression, number of classes for classification task
        :param class_task: bool, if False regression, else classification dataset
        :param scenario: int, scenario (1,2,4)
        :param mu_array: np.array (self.p) of means if regression, otherwise
        (self.K, self.p) array for classification
        :param sigma_array: np.array (self.p) of standard deviations, otherwise
        (self.K, self.p) array for classification
        :param regression_rule: only if self.class_task is False, we need a regression law
        
        self.p = p
        self.N_per_class = N_per_class
        self.K = K
        self.class_task = class_task
        self.scenario = scenario
        self.mu_array = mu_array
        self.sigma_array = sigma_array

        self.N = self.N_per_class * self.K  # total number of training data
        self.X = None  # input data, wo any additional dimensions
        self.y = None  # output labels

        self.minimal_dataset = False  # turns True once we generate the data
        self.X_transf = None  # transformed dataset (scenario 1, 2, 4)
        self.regression_rule = None  # array if we want to perform a regression task
        dct_kwargs = kwargs  # dictionary of args

        if not class_task and 'regression_rule' not in dct_kwargs.keys():
            raise ValueError("Regression law not provided")

        if 'path_minimal_data' not in dct_kwargs.keys():
            dct_kwargs['path_minimal_data'] = None
        else:
            self.A = np.load(join(dct_kwargs['path_minimal_data'], 'A.npy'))

            self.N = np.load(join(dct_kwargs['path_minimal_data'], 'N.npy'))
            self.X = np.load(join(dct_kwargs['path_minimal_data'], 'X.npy'))
            self.y = np.load(join(dct_kwargs['path_minimal_data'], 'y.npy'))

         self.generate_minimal_data()  # we generate the minimal dataset given the hyper-parameters
        if scenario is not None:
            if 'add_p' not in dct_kwargs.keys():
                raise ValueError("Specify the amount of additional features through add_p")
            elif dct_kwargs['add_p'] == 0:
                self.X_transf = self.X
                return
            self.add_p = dct_kwargs['add_p']

        if self.scenario == 1:  # additional noisy features
            self.noise_mean = np.array(dct_kwargs['noise_mean']) if dct_kwargs['noise_mean'] is not None else 0
            self.noise_sigma = np.array(dct_kwargs['noise_sigma']) if dct_kwargs['noise_sigma'] is not None else 1
            self.add_gaussian_noise()

        elif self.scenario == 2:  # additional redundant features
            self.A = np.array(dct_kwargs['A']) if 'A' in dct_kwargs.keys() else None
            self.add_redundancy()

        elif self.scenario == 4:  # add noise and redundancy
            self.redundancy_amount = dct_kwargs[
                'redundancy_amount'] if 'redundancy_amount' in dct_kwargs.keys() else 0.5
            self.A = np.array(dct_kwargs['A']) if 'A' in dct_kwargs.keys() else None
            self.noise_mean = np.array(dct_kwargs['noise_mean']) if dct_kwargs['noise_mean'] is not None else 0
            self.noise_sigma = np.array(dct_kwargs['noise_sigma']) if dct_kwargs['noise_sigma'] is not None else 1

            self.add_mixture()
        

    def generate_minimal_data(self):
        
        Here we generate the data by using the relevant features only.
        Each feature is Gaussian distributed. Mean and standard
        deviation for each variable varies depending on the user specification.

        The generic i-th feature is x_i
                    x_i = mean_i + N(0,1) * std_i, x_i in R^n_samples

        The labels are generating depending on the learning task.
        If class_task is False, a regression rule is needed, and
            y_k = np.dot(regression_rule, x_k)  , x_k in R^n_features,

        If class_task is True, the classifier the two distribution are
        given different values. # at the moment we are not considering the
        multi-classification task.
        
        if not self.class_task and self.regression_rule is None:  # for regression task
            raise ValueError("You need a learning rule to generate the regression law.")

        # regression task
        if not self.class_task:
            self.regression_rule = np.squeeze(self.regression_rule)

            if self.regression_rule.ndim == 1:  #  p or K are equal to one
                if self.regression_rule.size != self.p and self.regression_rule.size != self.K:
                    raise ValueError("The regression rule does not match the data dimensionality")
            else:  # if the dimensionality if bigger than one
                check_output, check_input = self.regression_rule.shape
                if check_output != self.K or check_input != self.p:
                    raise ValueError("The dimensions for the regression rule do not match the data")

            X_ = np.random.randn(self.p, self.N)  # we generate the new dataset
            for id_, (mu_, sigma_) in enumerate(zip(self.mu_array, self.sigma_array)):
                X_[id_] *= sigma_
                X_[id_] += mu_
            self.X = X_
            if self.K != 1 and self.p == 1:  # if the number of output != 1
                self.regression_rule = self.regression_rule.reshape(-1, 1)
            elif self.K == 1 and self.p != 1:
                self.regression_rule = self.regression_rule.reshape(-1, )
            self.y = np.dot(self.regression_rule, self.X)

        # classification task
        else:
            check_output_mu, check_input_mu = np.squeeze(self.mu_array).shape
            check_output_st, check_input_st = np.squeeze(self.sigma_array).shape

            if check_output_mu != self.K or check_output_st != self.K:
                raise ValueError("Arrays inconsistent with the number of classes")

            X_ = np.zeros((self.p, self.N))
            y_ = np.zeros((self.K, self.N))
            for k_, (mu_class_, sigma_class_) in enumerate(zip(self.mu_array, self.sigma_array)):  # for each class
                first_ = k_ * self.N_per_class  # n_per_class
                last_ = self.N if k_ == self.K - 1 else (k_ + 1) * self.N_per_class
                for id_, (mu_, sigma_) in enumerate(zip(mu_class_, sigma_class_)):
                    X_[id_, first_:last_] = mu_ + np.random.randn(last_ - first_) * sigma_
                y_[k_, first_:last_] = 1

            self.y = y_
            self.X = X_
        self.minimal_dataset = True

        return self

    def add_redundancy(self):
         We add redundancy to the dataset.
        Using a linear combination of the input features.
        
        if not self.minimal_dataset:
            raise ValueError("Generate the dataset first")

        if self.A is not None:
            check_add_feat, check_d = self.A.shape  # check the dimensions
            if check_d != self.p:
                raise ValueError("The dimension of the transformation does not check the amount of features")

        else:
            self.A = np.random.randn(self.add_p, self.p)  # add redundancy through a linear transformation

        self.X_transf = np.vstack((self.X, self.A.dot(self.X)))  #  save as an attribute the feature

    def add_gaussian_noise(self):
         We add noisy features to the dataset.
        This is done by adding Gaussian distributed
        random variables to the original features.
        
        if not self.minimal_dataset:
            raise ValueError("Generate the dataset first")

        if self.noise_mean.size == 1 and self.noise_sigma.size == 1:
            self.X_transf = np.vstack((self.X, (self.noise_mean +
                                                self.noise_sigma * np.random.randn(self.add_p, self.N))))
            return

        if self.add_p != self.noise_mean.size and self.add_p != self.noise_sigma.size:
            raise ValueError("Mismatch in dimensions")

        noise_feat = np.zeros((self.add_p, self.N))
        for i_, (noise_m_, noise_s_) in enumerate(zip(self.noise_mean, self.noise_sigma)):
            noise_feat[i_] = noise_m_ + noise_s_ * np.random.randn(self.N)

        self.X_transf = np.vstack((self.X, noise_feat))

        return self

    def add_mixture(self):
        With this call we add a percentage of redundancy and a (1-percentage) of noisy features.
        
        if self.A is None and np.isscalar(self.noise_mean) and np.isscalar(self.noise_sigma):
            n_noise_feat = (self.add_p - int(self.add_p * self.redundancy_amount))
            n_rdndt_feat = int(self.add_p * self.redundancy_amount)
            self.A = np.random.randn(n_rdndt_feat, self.p)
            noise_feat = self.noise_mean + self.noise_sigma * np.random.randn(n_noise_feat, self.N)

        elif self.A is not None:
            n_rdndt_feat, p_tmp = self.A.shape
            n_noise_feat = self.add_p - n_rdndt_feat
            if p_tmp != self.p:
                raise ValueError("Mismatch in dimensions for the linear transformation")

            if self.noise_mean.size != 1 and self.noise_sigma.size != 1:
                if self.noise_mean.size != self.noise_sigma.size:  # check that the shapes are consistent
                    raise ValueError("The noise values are inconsistent in shape")
                n_noise_feat = self.noise_mean.size

                # check not to exceed the add_p value
                if self.add_p != (n_noise_feat + n_rdndt_feat):
                    raise ValueError(
                        "The sum of the two dimensions is different from the specified number of additional features")
                noise_feat = np.zeros((self.noise_mean.size, self.N))
                for i_, (noise_m_, noise_s_) in enumerate(zip(self.noise_mean, self.noise_sigma)):
                    noise_feat[i_] = noise_m_ + noise_s_ * np.random.randn(self.N)
            else:
                noise_feat = self.noise_mean + self.noise_sigma * np.random.randn(n_noise_feat, self.N)

        X_ = np.vstack((self.X, noise_feat))
        self.X_transf = np.vstack((X_, np.dot(self.A, self.X)))
    """