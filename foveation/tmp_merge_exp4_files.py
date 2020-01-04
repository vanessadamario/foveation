import os
from os.path import join
import numpy as np

folder_root = '/om/user/vanessad/foveation/modified_std_small_MNIST_dataset'
folder_list = ['exp_4_tr', 'exp_4_ts']
edge_list = [0, 4, 6, 14, 26, 66]

for tmp_f_ in folder_list:
    f_ = join(folder_root, tmp_f_)
    for e_ in edge_list:
        for i in range(100):
            tmp = np.load(join(f_, 'exp_4_dim_%i_split_%i_tr.npy' % (e_, i)))
            if i == 0:
                x = tmp
            else:
                x = np.vstack((x, tmp))
        np.save(join(os.path.dirname(f_), 'exp_4_dim_%i_tr.npy' % (28 + e_*2)), x)
