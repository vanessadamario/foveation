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

# this is for repetitions
parser.add_argument('--repetition_folder_path', type=str, required=False)
FLAGS = parser.parse_args()

# where to save and retrieve the experiments
output_path = {
    'om': '/om/user/vanessad/IMDb_framework/results',
    'vanessa': '/Users/vanessa/Desktop/test_IMDb'}[FLAGS.host_filesystem]

output_path = join(output_path,
                   'repetition_' + FLAGS.repetition_folder_path + '/')

os.makedirs(output_path, exist_ok=True)

if FLAGS.offset_index is None:
    FLAGS.offset_index = 0
if FLAGS.check_train is None:
    FLAGS.check_train = False

def generate_experiments(id):
    """ Generation of the experiments. """
    experiments.generate_experiments(output_path)


def run_train(id):
    """ Run the experiments. """
    from runs.train import check_and_train
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    check_and_train(opt, output_path)


def remove_id(id):
    from runs import remove_id as remove
    remove.run(id, output_path)


def update_json(id):
    from runs.update import check_update
    """ Write on the json if the experiments are completed,o
    by changing the flag. """
    check_update(output_path)


switcher = {
    'train': run_train,
    'gen': generate_experiments,
    'remove': remove_id,
    'update': update_json
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)