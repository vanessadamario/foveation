import os
import argparse
from os.path import join
from runs import experiments

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--offset_index', type=int, required=False)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--check_train', type=bool, required=False)
parser.add_argument('--dataset_name', type=str, required=False)

# this is for repetitions
parser.add_argument('--repetition_folder_path', type=str, required=False)
FLAGS = parser.parse_args()

# where to save and retrieve the experiments
output_path = {
    'om': '/om/user/vanessad/synthetic_framework/results',
    'om2': '/om2/user/vanessad/synthetic_framework/results',
    'vanessa': '/Users/vanessa/Desktop/test'}[FLAGS.host_filesystem]

output_path = join(output_path, 'repetition_' + FLAGS.repetition_folder_path + '/')
dataset_path = (output_path + FLAGS.dataset_name)

os.makedirs(output_path, exist_ok=True)

if FLAGS.offset_index is None:
    FLAGS.offset_index = 0
if FLAGS.check_train is None:
    FLAGS.check_train = False

def generate_data(id):
    """ here we generate the data """
    from runs.generate_data import DatasetGenerator
    os.makedirs(dataset_path, exist_ok=False)

    DatasetGenerator(data_path=dataset_path,
                     load=False,
                     key_dataset=FLAGS.dataset_name)

def generate_experiments(id):
    # DONE
    """ Generation of the experiments. """
    experiments.generate_experiments(output_path)


def run_train(id):
    """ Run the experiments. """
    # DONE
    from runs.train import check_and_train
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    check_and_train(opt, output_path)


def remove_id(id):
    from runs import remove_id as remove
    remove.run(id, output_path)


def update_json(id):
    # DONE
    from runs.update import check_update
    """ Write on the json if the experiments are completed,
    by changing the flag. """ofg
    check_update(output_path)


switcher = {
    'train': run_train,
    'dataset': generate_data,
    'gen': generate_experiments,
    'remove': remove_id,
    'update': update_json
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)