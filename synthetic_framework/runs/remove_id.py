import json
import shutil
from runs.experiments import decode_exp


def run(id, output_path):
    """
    :param id: the id for the experiment
    :param output_path: the path where we can find the json
    """
    # TODO: do we want to remove the folder here, if it exists?
    info_path = output_path + 'train.json'
    with open(info_path) as infile:
        info = json.load(infile)

    found = False
    for x in info:
        if info[str(x)]['id'] == id:
            exp = decode_exp(info[str(x)])
            print("removing id {}: {}".format(id, info[str(x)]))
            del info[str(x)]
            if exp.train_completed:
                shutil.rmtree(exp.output_path)
            found = True
            break

    if not found:
        print("id {} not found, it may have been deleted or never existed".format(id))

    with open(info_path, 'w') as outfile:
        json.dump(info, outfile)
