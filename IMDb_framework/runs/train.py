import os
from os.path import join
from runs.network import Network
from runs.CNN_text_classification import ShallowCNNTextClassifier
from runs.experiments import transf_dct


def train_network(opt):
    print('train')
    print('list imported from experiments.py', transf_dct)
    if opt.hyper.architecture == 'CNN':
        ShallowCNNTextClassifier(opt)
    else:
        Network(opt)


def check_and_train(opt, output_path):
    """ Check if the experiments has already been performed.
    If it is not, train otherwise retrieve the path relative to the experiment.
    :param opt: Experiment instance. It contains the output path for the experiment
    under study
    :param output_path: the output path for the *.json file. necessary to change the *.json
    at different time steps
    """
    if opt.train_completed:
        print("Object: ", opt)
        print("Experiment already trained in " + opt.output_path)
        return

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    train_network(opt)

    # we write an empty *.txt file with the completed experiment
    flag_completed_dir = join(output_path, 'flag_completed')
    os.makedirs(flag_completed_dir, exist_ok=True)
    file_object = open(join(flag_completed_dir, "complete_%s.txt" % str(opt.id)), "w")
    file_object.close()

