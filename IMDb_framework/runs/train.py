import os
import sys
from os.path import join
from runs.network import Network
from runs.generate_input_from_embedding import DataGenerator
from runs.CNN_text_classification import ShallowCNNTextClassifier


def train_network(id, opt, embedding_path):
    # check that this works properly
    print('do stuff')
    sys.stdout.flush()
    dataset_gen = DataGenerator(opt, n_max=1000, split="train", train=True, embedding_path=embedding_path)
    print('n training', opt.dataset.n_training)
    sys.stdout.flush()

    print('instances of class DataGenerator and Embedding created')
    sys.stdout.flush()

    x_tr, y_tr = dataset_gen.generate_x_y()
    print('sizes', len(x_tr), y_tr.shape)
    sys.stdout.flush()

    print('x_train generation, completed')
    sys.stdout.flush()

    print('training embedding, completed')
    sys.stdout.flush()
    dataset_gen.train = False
    dataset_gen.split = 'valid'
    x_vl, y_vl = dataset_gen.generate_x_y()
    print('x_valid generation, completed', len(x_vl))
    sys.stdout.flush()

    if opt.hyper.architecture == 'CNN':
        model = ShallowCNNTextClassifier(opt)
    else:
        model = Network(opt)
    model.optimize(x_tr, y_tr, x_vl, y_vl)
    del x_tr, x_vl

    dataset_gen.split = 'test'
    x_ts, y_ts = dataset_gen.generate_x_y()
    print('x_test generation, completed', len(x_ts))
    model.test_and_save(x_ts, y_ts)


def check_and_train(id, opt, output_path, embedding_path):
    """ Check if the experiments has already been performed.
    If it is not, train otherwise retrieve the path relative to the experiment.
    :param opt: Experiment instance. It contains the output path for the experiment
    under study
    :param output_path: the output path for the *.json file. necessary to change the *.json
    at different time steps
    """
    print('Im in check and train')
    sys.stdout.flush()

    if opt.train_completed:
        print("Object: ", opt)
        sys.stdout.flush()

        print("Experiment already trained in " + opt.output_path)
        sys.stdout.flush()

        return

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    print(opt)
    print('Im gonna train')
    sys.stdout.flush()

    train_network(id, opt, embedding_path)

    # we write an empty *.txt file with the completed experiment
    flag_completed_dir = join(output_path, 'flag_completed')
    os.makedirs(flag_completed_dir, exist_ok=True)
    file_object = open(join(flag_completed_dir, "complete_%s.txt" % str(opt.id)), "w")
    file_object.close()

