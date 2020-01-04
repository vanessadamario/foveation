# output_dimension = [28, 36, 40, 56, 80, 120, 160]
# scenario = [1, 2, 4]

import os
import numpy as np
import argparse
from os.path import join
from foveation.DATASET_GENERATOR import DatasetGenerator
from tensorflow.keras.datasets import mnist


parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=int, required=True)
parser.add_argument('--output_dimension', type=int, required=True)
FLAGS = parser.parse_args()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalization

n_splits = 100
folder_dataset = '/om/user/vanessad/foveation/standardized_MNIST_dataset'
os.makedirs(folder_dataset, exist_ok=True)

folder_scenario = join(folder_dataset, 'exp_%i' % parser[FLAGS.scenario])
os.makedirs(folder_scenario, exist_ok=True)

folder_dimension = join(folder_scenario, 'dim_%i' % parser[FLAGS.output_dimension])
os.makedirs(folder_dimension)

folder_train = join(folder_dimension, 'train')
folder_test = join(folder_dimension, 'test')

os.makedirs(folder_train)
os.makedirs(folder_test)


for kk, x_train_split in enumerate(np.split(x_train, n_splits, axis=0)):
    DG = DatasetGenerator(x_train_split,
                          output_dim=parser[FLAGS.output_dimension],
                          scenario=parser[FLAGS.scenario])
    DG.run()
    np.save(join(folder_train, 'split_%i.npy' % kk), DG.output)

for kk, x_test_split in enumerate(np.split(x_test, n_splits, axis=0)):
    DG = DatasetGenerator(x_test_split,
                          output_dim=parser[FLAGS.output_dimension],
                          scenario=parser[FLAGS.scenario])
    DG.run()
    np.save(join(folder_test, 'split_%i.npy' % kk), DG.output)